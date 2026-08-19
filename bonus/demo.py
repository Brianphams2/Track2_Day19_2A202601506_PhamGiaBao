"""Bonus Challenge Demo — 5-Query Showcase for HybridMemoryAgent.

Demonstrates:
  1. Direct episodic memory recall (Kubernetes reading notes).
  2. Profile-guided recommendation (using topic_affinity from Feature Store).
  3. Real-time activity context (using queries_last_hour / velocity features).
  4. Paraphrase semantic retrieval (vector similarity across Vietnamese phrasing).
  5. Mixed intent recall (combining episodic knowledge + profile constraints).

Run:
  python bonus/demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bonus.agent import HybridMemoryAgent


def seed_sample_memories(agent: HybridMemoryAgent, user_id: str = "u_001") -> None:
    """Pre-populate episodic memory for user u_001 with realistic technical reading notes."""
    notes = [
        (
            "Kubernetes Cluster Architecture: Ghi chú về control plane gồm API Server, etcd lưu trữ trạng thái, "
            "Kube-Scheduler phân phối pod và Controller Manager duy trì desired state. Worker nodes chạy kubelet và kube-proxy.",
            {"tag": "k8s_architecture", "doc_type": "note"},
        ),
        (
            "Triển khai Horizontal Pod Autoscaler (HPA) trên Kubernetes: HPA tự động tăng giảm số lượng pod dựa trên "
            "mức sử dụng CPU/Memory hoặc custom metrics từ Prometheus khi lưu lượng truy cập tăng vọt.",
            {"tag": "autoscaling", "doc_type": "guide"},
        ),
        (
            "Cloud Security Best Practices: Áp dụng nguyên tắc Least Privilege cho IAM roles, mã hóa dữ liệu at-rest bằng AWS KMS "
            "hoặc HashiCorp Vault, và cấu hình Network Security Groups / Security Groups chặn port không cần thiết.",
            {"tag": "cloud_security", "doc_type": "cheatsheet"},
        ),
        (
            "Database Replication & Consistency: Phân biệt Strong Consistency và Eventual Consistency theo định lý CAP. "
            "Sử dụng Raft consensus algorithm cho phân tán đồng thuận.",
            {"tag": "database", "doc_type": "note"},
        ),
        (
            "Tối ưu chi phí hạ tầng Cloud với Spot Instances: Tận dụng máy ảo dư thừa với giá rẻ hơn tới 70-90%, "
            "kết hợp graceful shutdown khi nhận thông báo preempt trong vòng 2 phút.",
            {"tag": "cost_optimization", "doc_type": "guide"},
        ),
    ]

    for text, meta in notes:
        agent.remember(text=text, user_id=user_id, metadata=meta)


def main() -> int:
    print("=" * 70)
    print("  BONUS CHALLENGE: Hybrid AI Memory Agent (Vector + Feature Store)")
    print("=" * 70)

    feast_path = REPO_ROOT / "app" / "feast_repo"
    agent = HybridMemoryAgent(feast_repo_path=feast_path)

    USER_ID = "u_001"
    print(f"\n[1] Seeding episodic memories for user '{USER_ID}'...")
    seed_sample_memories(agent, user_id=USER_ID)
    print("    -> 5 reading notes & guides ingested into Qdrant vector memory.")

    queries = [
        ("Query 1 (Episodic Direct Hit)", "Tôi đã đọc gì về Kubernetes?"),
        ("Query 2 (Profile Context)", "Recommend tài liệu kỹ thuật nào tôi nên đọc tiếp?"),
        ("Query 3 (Fresh Activity Context)", "Tôi đang quan tâm và tìm kiếm những gì gần đây?"),
        ("Query 4 (Paraphrase Vector Retrieval)", "Tài liệu về tự động mở rộng hạ tầng khi traffic tăng cao?"),
        ("Query 5 (Mixed Intent: Episodic + Profile)", "Cho tôi summary về cloud security và phân quyền IAM"),
    ]

    print("\n[2] Executing 5 Showcase Queries:\n")
    for title, q in queries:
        print(f"\n>>> {title}")
        context = agent.recall(query=q, user_id=USER_ID, top_k=2)
        print(context)
        print()

    print("=" * 70)
    print("  DEMO COMPLETE: All 5 queries recalled with assembled hybrid context.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
