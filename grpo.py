import dataclasses
import gc
import math
from collections import defaultdict
from typing import Callable, List

import numpy as np
import torch

from data_types import Episode, MiniBatch
from qwen2_model import Transformer
from tokenizer import Tokenizer


@torch.no_grad()
def rollout(
    model: Transformer,  # 待推理的Transformer模型
    batch: MiniBatch, # 输入数据批次（包含多个prompt）
    tokenizer: Tokenizer, # 分词器
    max_gen_len: int, # 最大生成长度
    num_answer_per_question: int, # 每个prompt生成多少个答案
    reward_function: Callable, # 计算奖励的函数
    device: torch.device, # 计算设备（CPU/GPU）
    dtype: torch.dtype, # 计算数据类型（float16/float32）
) -> List[Episode]: # 返回生成的Episode列表
    """
    Rollout阶段：生成答案
    1. 初始化KV缓存
    2. 初始化tokens
    3. 遍历每个位置
    4. 计算logits
    5. 采样下一个token
    6. 更新tokens
    """
    end_token = tokenizer.eos_token # 结束标记
    end_token_id = tokenizer.eos_token_id # 结束标记ID
    pad_token_id = tokenizer.pad_token_id # 填充标记ID
    prefix_token_ids = batch.prefix_token_ids # 每个prompt的输入token ID
    bsz = len(batch.prefix) * num_answer_per_question # 总batch大小
    min_prompt_len = min(len(t) for t in prefix_token_ids) # 最短prompt长度
    max_prompt_len = max(len(t) for t in prefix_token_ids) # 最长prompt长度
    total_len = max_gen_len + max_prompt_len # 总长度
    model.init_kv_cache( # 初始化KV缓存
        max_batch_size=bsz, # 最大batch大小
        max_seq_len=total_len, # 最大序列长度
        device=device, # 设备
        dtype=dtype, # 数据类型
    )
    tokens = torch.full((bsz, total_len), pad_token_id, dtype=torch.long, device=device) # 初始化tokens
    for k, t in enumerate(prefix_token_ids): # 遍历每个prompt
        offset = k * num_answer_per_question # 偏移量
        for i in range(num_answer_per_question): # 遍历每个答案
            tokens[offset + i, : len(t)] = torch.tensor( # 填充tokens
                t, dtype=torch.long, device=device # 数据类型和设备
            )

    prev_pos = 0 # 前一个位置
    input_text_mask = tokens != pad_token_id # 输入文本掩码
    assert min_prompt_len < total_len # 确保最小prompt长度小于总长度
    is_finished = torch.zeros((bsz,), dtype=torch.bool, device=device) # 初始化完成标记

    for cur_pos in range(min_prompt_len, total_len): # 遍历每个位置
        print(
            f"\r* Generating trajectories: {cur_pos-min_prompt_len:>4d}/{total_len-min_prompt_len:>4d}", # 打印进度
            flush=True,
            end="",
        )
        with torch.autocast(device_type=device.type, dtype=dtype): # 自动混合精度
            logits = model.inference(tokens[:, prev_pos:cur_pos], prev_pos) # 推理
        probs = torch.softmax(logits[:, -1], dim=-1) # 计算概率
        next_token = torch.multinomial(probs, num_samples=1) # 采样下一个token
        next_token = next_token.reshape(-1) # 重塑形状
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token # 如果输入文本掩码为True，则使用输入文本，否则使用下一个token
        )
        # if an rollout is finished, we fill the rest of the tokens with pad_token_id
        next_token = torch.where(is_finished, pad_token_id, next_token) # 如果完成标记为True，则使用填充token，否则使用下一个token
        tokens[:, cur_pos] = next_token # 更新tokens
        if end_token_id is not None: # 如果结束标记ID不为空
            is_end_token = next_token == end_token_id # 判断是否为结束标记
            is_generated_token = ~input_text_mask[:, cur_pos] # 判断是否为生成标记
            is_finished = is_finished | (is_end_token & is_generated_token) # 更新完成标记
        prev_pos = cur_pos # 更新前一个位置
        if is_finished.all(): # 如果所有轨迹都完成
            break
    model.del_kv_cache() # 删除KV缓存
    gc.collect() # 回收垃圾
    torch.cuda.empty_cache() # 清空缓存
    is_finished_list = is_finished.tolist() # 转换为列表
    tokens_list = tokens.tolist() # 转换为列表

    # prepare the output episodes
    episodes = [] # 初始化Episode列表   
    for i in range(bsz // num_answer_per_question): # 遍历每个prompt
        for j in range(num_answer_per_question): # 遍历每个答案
            idx = i * num_answer_per_question + j # 索引
            generated_token_ids = tokens_list[idx][len(batch.prefix_token_ids[i]) :] # 生成token ID
            # remove padding tokens
            if pad_token_id in generated_token_ids: # 如果填充token ID在生成token ID中
                generated_token_ids = generated_token_ids[
                    : generated_token_ids.index(pad_token_id) # 删除填充token ID
                ]
            generated_text = tokenizer.detokenize(generated_token_ids) # 反向分词
            rewards = reward_function(
                response=generated_text, # 响应
                numbers=batch.numbers[i], # 数字
                target=batch.target[i], # 目标
                end_token=end_token, # 结束标记
            )
            episode = Episode(
                prefix=batch.prefix[i], # 前缀
                text=batch.prefix[i] + generated_text, # 文本
                prefix_token_ids=batch.prefix_token_ids[i], # 前缀token ID
                prefix_tokens=batch.prefix_tokens[i], # 前缀tokens
                generated_token_ids=generated_token_ids, # 生成token ID
                is_finished=is_finished_list[idx], # 完成标记
                reward=rewards["reward"], # 奖励
                reward_info=rewards["reward_info"], # 奖励信息
            )
            episodes.append(episode) # 添加到Episode列表
    # clear the output line
    print("\r", end=" " * 100, flush=True) # 清空输出行
    return episodes # 返回Episode列表


def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]: # 归一化奖励
    """Normalize rewards per group. A group is defined by the prefix."""
    groups = defaultdict(list) # 初始化组
    for episode in episodes: # 遍历每个Episode
        groups[tuple(episode.prefix)].append(episode) # 添加到组
    output = [] # 初始化输出
    for group in groups.values():
        group_rewards = [item.reward for item in group] # 组奖励
        mean_reward = np.mean(group_rewards) # 平均奖励
        std_reward = np.std(group_rewards) # 标准差
        for episode in group: # 遍历每个Episode
            normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-4) # 归一化奖励
            episode = dataclasses.replace(episode, reward=normalized_reward) # 替换奖励
            output.append(episode) # 添加到输出
    return output # 返回输出


def compute_entropy(logits: torch.Tensor) -> torch.Tensor: # 计算熵
    probs = torch.nn.functional.softmax(logits, dim=-1) # 计算概率
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1) # 计算熵
    return entropy # 返回熵


def update_policy(
    model: Transformer, # 待更新的Transformer模型
    optimizer: torch.optim.Optimizer, # 优化器
    episodes: List[Episode], # 轨迹列表
    micro_batch_size: int, # 微批大小
    pad_token_id: int, # 填充token ID
    max_grad_norm: float, # 最大梯度范数
    device: torch.device, # 计算设备（CPU/GPU）
    dtype: torch.dtype, # 计算数据类型（float16/float32）
):
    """Update the policy using the GRPO algorithm."""
    episodes = normalize_rewards_per_group(episodes) # 归一化奖励
    # sort episodes by token length for efficient (micro-)batching
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids)) # 按token长度排序
    num_micro_batches = math.ceil(len(episodes) / micro_batch_size) # 计算微批数量
    num_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes) # 计算目标token数量
    entropy = 0.0 # 初始化熵

    for i in range(0, len(episodes), micro_batch_size): # 遍历每个微批
        print(
            f"\r* Computing policy gradient: {i:>2d}/{len(episodes):>2d}", # 打印进度
            flush=True,
            end="",
        )
        j = min(i + micro_batch_size, len(episodes)) # 计算当前微批的结束索引
        batch_episodes = episodes[i:j] # 获取当前微批的轨迹
        batch_lengths = [
            len(episode.prefix_token_ids) + len(episode.generated_token_ids) # 计算当前微批的轨迹长度
            for episode in batch_episodes
        ] # 计算当前微批的轨迹长度
        batch_max_length = max(batch_lengths) # 计算当前微批的最大长度
        batch_token_ids = [
            episode.prefix_token_ids # 获取当前微批的轨迹token ID
            + episode.generated_token_ids # 获取当前微批的生成token ID
            + [pad_token_id] * (batch_max_length - batch_lengths[i]) # 填充当前微批的轨迹token ID
            for i, episode in enumerate(batch_episodes) # 遍历当前微批的轨迹
        ]
        batch_masks = [
            [0] * len(episode.prefix_token_ids) # 填充当前微批的轨迹token ID
            + [1] * len(episode.generated_token_ids) # 填充当前微批的生成token ID
            + [0] * (batch_max_length - batch_lengths[i]) # 填充当前微批的轨迹token ID
            for i, episode in enumerate(batch_episodes) # 遍历当前微批的轨迹
        ]
        batch_advantages = [episode.reward for episode in batch_episodes] # 计算当前微批的奖励
        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long) # 转换为张量
        batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool) # 转换为张量
        batch_advantages = torch.tensor(
            batch_advantages, device=device, dtype=torch.float32 # 转换为张量
        )

        with torch.autocast(device_type=device.type, dtype=dtype): # 自动混合精度
            input_token_ids = batch_token_ids[:, :-1] # 获取当前微批的输入token ID
            target_token_ids = batch_token_ids[:, 1:] # 获取当前微批的目标token ID
            target_masks = batch_masks[:, 1:] # 获取当前微批的目标掩码
            logits = model.forward(input_token_ids).float() # 获取当前微批的logits

        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), # 重塑形状
            target_token_ids.reshape(-1), # 重塑形状
            ignore_index=pad_token_id, # 忽略填充token ID
            reduction="none", # 不减少维度
        ).reshape(input_token_ids.shape[0], -1) # 重塑形状

        with torch.no_grad(): # 禁用梯度计算
            token_entropy = compute_entropy(logits) # 计算熵
            entropy = entropy + (token_entropy * target_masks).sum() / num_target_tokens # 更新熵

        obj = log_probs * batch_advantages[:, None] # 计算目标
        # per-token objective
        obj = (obj * target_masks).sum() / num_target_tokens # 计算目标
        loss = -obj # 计算损失
        loss.backward() # 计算梯度

    # update the policy
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm # 计算梯度范数
    )
    optimizer.step() # 更新参数
    optimizer.zero_grad(set_to_none=True) # 清空梯度
    # 返回结果
    return {
        "loss": loss.item(), # 损失
        "grad_norm": grad_norm.item(), # 梯度范数
        "entropy": entropy.item(), # 熵
    }
