"""Bluetooth PCAP RAG + LLM demo.

This script:
1) Parses a PCAP into a simple RAG database stored as a pandas DataFrame.
2) Retrieves relevant packets by keyword overlap.
3) Calls an OpenAI model with the retrieved context.

Usage:
  python usecase/bluetooth_rag_demo.py build --pcap data/example.pcap --out rag.csv
  python usecase/bluetooth_rag_demo.py ask --rag rag.csv --question "Which devices are advertising?"
  python usecase/bluetooth_rag_demo.py demo --question "What channels show advertising traffic?"
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd
from openai import OpenAI
from scapy.all import rdpcap
from scapy.layers.bluetooth import BTLE, BTLE_ADV, BTLE_ADV_IND


@dataclass
class RagConfig:
    max_packets: int = 2000
    max_chars: int = 900


def _packet_layers(packet) -> List[str]:
    return [layer.__name__ for layer in packet.layers()]


def _packet_text(packet, index: int) -> str:
    layers = _packet_layers(packet)
    summary = packet.summary()
    btle_fields = []
    if packet.haslayer(BTLE):
        btle = packet.getlayer(BTLE)
        btle_fields.append(f"access_address={getattr(btle, 'access_addr', 'n/a')}")
        btle_fields.append(f"channel={getattr(btle, 'channel', 'n/a')}")
    if packet.haslayer(BTLE_ADV):
        adv = packet.getlayer(BTLE_ADV)
        btle_fields.append(f"adv_address={getattr(adv, 'AdvA', 'n/a')}")
        btle_fields.append(f"adv_type={adv.__class__.__name__}")
    if packet.haslayer(BTLE_ADV_IND):
        adv_ind = packet.getlayer(BTLE_ADV_IND)
        btle_fields.append(f"addr={getattr(adv_ind, 'AdvA', 'n/a')}")
    btle_detail = "; ".join(btle_fields) if btle_fields else "no-btle-fields"
    return (
        f"packet_index={index}; layers={','.join(layers)}; "
        f"summary={summary}; btle={btle_detail}"
    )


def pcap_to_rag_dataframe(pcap_path: str, config: RagConfig) -> pd.DataFrame:
    packets = rdpcap(pcap_path)
    rows = []
    for index, packet in enumerate(packets[: config.max_packets], start=1):
        timestamp = getattr(packet, "time", None)
        text = _packet_text(packet, index)
        rows.append(
            {
                "packet_index": index,
                "timestamp": timestamp,
                "layers": ",".join(_packet_layers(packet)),
                "summary": packet.summary(),
                "text": text[: config.max_chars],
            }
        )
    return pd.DataFrame(rows)


def _tokenize(text: str) -> List[str]:
    return [token for token in "".join(
        [char.lower() if char.isalnum() else " " for char in text]
    ).split() if token]


def retrieve_context(df: pd.DataFrame, question: str, top_k: int = 5) -> pd.DataFrame:
    question_tokens = set(_tokenize(question))

    def score_row(row_text: str) -> int:
        row_tokens = set(_tokenize(row_text))
        return len(question_tokens & row_tokens)

    scored = df.copy()
    scored["score"] = scored["text"].apply(score_row)
    scored = scored.sort_values(by="score", ascending=False)
    return scored.head(top_k)


def build_prompt(question: str, context_rows: Iterable[dict]) -> str:
    context_lines = [f"- {row['text']}" for row in context_rows]
    context_block = "\n".join(context_lines)
    return (
        "You are a Bluetooth capture analyst. Use the packet context to answer the question.\n"
        "If the context is insufficient, say what is missing.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context_block}"
    )


def ask_openai(question: str, context_rows: Iterable[dict], model: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)
    prompt = build_prompt(question, context_rows)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You answer questions about Bluetooth PCAPs."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def run_build(args: argparse.Namespace) -> None:
    config = RagConfig(max_packets=args.max_packets)
    df = pcap_to_rag_dataframe(args.pcap, config)
    df.to_csv(args.out, index=False)
    print(f"Saved RAG database with {len(df)} packets to {args.out}")


def run_ask(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.rag)
    context = retrieve_context(df, args.question, top_k=args.top_k)
    answer = ask_openai(args.question, context.to_dict("records"), args.model)
    print("Answer:\n", answer)


def _demo_dataframe() -> pd.DataFrame:
    now = time.time()
    demo_rows = [
        {
            "packet_index": 1,
            "timestamp": now,
            "layers": "BTLE,BTLE_ADV,BTLE_ADV_IND",
            "summary": "BTLE ADV_IND",
            "text": "packet_index=1; layers=BTLE,BTLE_ADV,BTLE_ADV_IND; "
            "summary=BTLE ADV_IND; btle=access_address=0x8e89bed6; "
            "channel=37; adv_address=aa:bb:cc:dd:ee:ff; adv_type=BTLE_ADV_IND",
        },
        {
            "packet_index": 2,
            "timestamp": now + 0.01,
            "layers": "BTLE,BTLE_ADV",
            "summary": "BTLE ADV_NONCONN_IND",
            "text": "packet_index=2; layers=BTLE,BTLE_ADV; "
            "summary=BTLE ADV_NONCONN_IND; btle=access_address=0x8e89bed6; "
            "channel=38; adv_address=11:22:33:44:55:66; adv_type=BTLE_ADV",
        },
        {
            "packet_index": 3,
            "timestamp": now + 0.02,
            "layers": "BTLE",
            "summary": "BTLE DATA",
            "text": "packet_index=3; layers=BTLE; summary=BTLE DATA; "
            "btle=access_address=0x8e89bed6; channel=24; no-btle-fields",
        },
    ]
    return pd.DataFrame(demo_rows)


def run_demo(args: argparse.Namespace) -> None:
    df = _demo_dataframe()
    context = retrieve_context(df, args.question, top_k=args.top_k)
    answer = ask_openai(args.question, context.to_dict("records"), args.model)
    print("Answer:\n", answer)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bluetooth PCAP RAG + LLM demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build RAG CSV from PCAP")
    build_parser.add_argument("--pcap", required=True, help="Path to PCAP file")
    build_parser.add_argument("--out", default="bluetooth_rag.csv", help="Output CSV path")
    build_parser.add_argument("--max-packets", type=int, default=2000, help="Packet limit")
    build_parser.set_defaults(func=run_build)

    ask_parser = subparsers.add_parser("ask", help="Ask a question using RAG CSV")
    ask_parser.add_argument("--rag", required=True, help="Path to RAG CSV")
    ask_parser.add_argument("--question", required=True, help="Question to ask")
    ask_parser.add_argument("--top-k", type=int, default=5, help="Top K context packets")
    ask_parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    ask_parser.set_defaults(func=run_ask)

    demo_parser = subparsers.add_parser("demo", help="Run demo using sample Bluetooth packets")
    demo_parser.add_argument("--question", required=True, help="Question to ask")
    demo_parser.add_argument("--top-k", type=int, default=5, help="Top K context packets")
    demo_parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    demo_parser.set_defaults(func=run_demo)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
