#!/usr/bin/env python3
# qwen3_token_id_to_token.py
# 用途：将 Qwen3 的 token id 映射回 token（子词）与完整文本
# Usage:
#   python qwen3_token_id_to_token.py --ids 128001
#   python qwen3_token_id_to_token.py --ids 128001,128007,320 --keep-special
#   python qwen3_token_id_to_token.py --model Qwen/Qwen3-0.5B --ids 123 456 789

from __future__ import annotations
import argparse
from typing import Iterable, List, Union

# 可选依赖支持 numpy/torch 张量输入（若未安装则自动忽略）
try:
    import numpy as np  # type: ignore
except Exception:
    np = None
try:
    import torch  # type: ignore
except Exception:
    torch = None

from transformers import AutoTokenizer


def _to_id_list(
    ids: Union[int, Iterable[int], "np.ndarray", "torch.Tensor"]
) -> List[int]:
    """将各种形式的 ids 统一转成 Python 的 List[int]。"""
    if isinstance(ids, int):
        return [int(ids)]
    if np is not None and isinstance(ids, np.ndarray):
        return [int(x) for x in ids.astype(int).tolist()]
    if torch is not None and isinstance(ids, torch.Tensor):
        return [int(x) for x in ids.detach().cpu().tolist()]
    # 一般可迭代
    return [int(x) for x in ids]


def _visible(s: str) -> str:
    """让不可见字符（换行/制表/回车/空格）更直观。"""
    return (
        s.replace("\\", "\\\\")
         .replace("\n", "\\n")
         .replace("\r", "\\r")
         .replace("\t", "\\t")
    )


def load_qwen3_tokenizer(model_name_or_path: str = "Qwen/Qwen3-0.5B-Instruct"):
    """
    加载 Qwen3 分词器。
    某些 Qwen3 版本需要 trust_remote_code=True 才能正确加载。
    """
    tok = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    return tok


def ids_to_tokens_and_text(
    ids: Union[int, Iterable[int], "np.ndarray", "torch.Tensor"],
    tokenizer=None,
    model_name_or_path: str = "Qwen/Qwen3-0.5B-Instruct",
    skip_special_tokens: bool = False,
    make_visible: bool = True,
):
    """
    将 token ids 映射回 token（子词）列表与整体文本。
    返回: dict = {
        "ids": [...],
        "tokens": [...],           # 每个 id 对应的字符串 token（包含/不包含特殊符号取决于 skip_special_tokens）
        "text": "..."              # decode 后的整体文本
    }
    """
    if tokenizer is None:
        tokenizer = load_qwen3_tokenizer(model_name_or_path)

    id_list = _to_id_list(ids)

    # 逐 token 映射
    # convert_ids_to_tokens 不会合并子词；是否包含特殊符号由 fast tokenizer 的实现决定
    tokens = tokenizer.convert_ids_to_tokens(id_list)

    # 整体解码
    text = tokenizer.decode(
        id_list,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )

    if make_visible:
        tokens = [_visible(t) for t in tokens]
        text = _visible(text)

    return {
        "ids": id_list,
        "tokens": tokens,
        "text": text,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3 token id → token / decoded text"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.5B-Instruct",
        help="Hugging Face 上的模型名或本地 tokenizer 路径（默认 Qwen/Qwen3-0.5B-Instruct）"
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        help="token id 列表。可用空格分隔：--ids 128001 320；也可用逗号：--ids 128001,320"
    )
    parser.add_argument(
        "--keep-special",
        action="store_true",
        help="decode 时保留特殊 token（默认跳过特殊 token）"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="不做可见化转义（\\n、\\t 等原样展示）"
    )
    args = parser.parse_args()

    if not args.ids:
        parser.error("请通过 --ids 提供一个或多个 token id，例如：--ids 128001 或 --ids 128001,320")

    # 将空格或逗号混合的输入统一解析
    flat: List[str] = []
    for part in args.ids:
        flat.extend(x for x in part.split(",") if x != "")
    id_list = [int(x) for x in flat]

    tokenizer = load_qwen3_tokenizer(args.model)
    out = ids_to_tokens_and_text(
        id_list,
        tokenizer=tokenizer,
        skip_special_tokens=not args.keep_special,
        make_visible=not args.raw,
    )

    print("# IDs:")
    print(out["ids"])
    print("# Tokens:")
    print(out["tokens"])
    print("# Decoded text:")
    print(out["text"])


if __name__ == "__main__":
    main()