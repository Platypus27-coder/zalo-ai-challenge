"""Generate high-resolution (300 DPI) Unified System Architecture Diagram PNG.

Creates a publication-quality diagram covering:
1. SFT / QLoRA Training Pipeline
2. Input & Speech Processing (Part B: ASR & Normalizer)
3. Agent LLM Core (Part C: Prompt, Tool Filter, Qwen3-4B + LoRA)
4. Guardrails (Task Tracker, ID Interceptor, Evidence Ledger)
5. Simulated Tool Environment (Part A: zaic_tools.ToolEnv)
6. State Hash, Linter & Submission Checkpointing
"""
import os
import sys
from pathlib import Path

# Set Matplotlib cache dir to prevent warnings
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent / ".mpl_cache")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw_box(ax, x, y, w, h, title, subtitle, bg_color, border_color, title_color="#0F172A", subtitle_color="#475569", radius=0.015):
    """Draw a modern rounded card with title and subtitle."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=1.8,
        zorder=3
    )
    ax.add_patch(box)
    
    # Text
    if subtitle:
        ax.text(
            x + w / 2, y + h * 0.65, title,
            ha="center", va="center",
            fontsize=9.5, fontweight="bold",
            color=title_color, family="Segoe UI",
            zorder=4
        )
        ax.text(
            x + w / 2, y + h * 0.32, subtitle,
            ha="center", va="center",
            fontsize=7.8, fontweight="normal",
            color=subtitle_color, family="Segoe UI",
            zorder=4, multialignment="center"
        )
    else:
        ax.text(
            x + w / 2, y + h * 0.5, title,
            ha="center", va="center",
            fontsize=10.5, fontweight="bold",
            color=title_color, family="Segoe UI",
            zorder=4
        )


def draw_arrow(ax, p1, p2, text="", color="#475569", rad=0.0, text_offset=(0, 0), linestyle="-", width=1.8):
    """Draw a smooth directed arrow with optional label."""
    style = f"arc3,rad={rad}"
    arrow = FancyArrowPatch(
        p1, p2,
        connectionstyle=style,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=width,
        color=color,
        linestyle=linestyle,
        zorder=5
    )
    ax.add_patch(arrow)
    
    if text:
        mx = (p1[0] + p2[0]) / 2 + text_offset[0]
        my = (p1[1] + p2[1]) / 2 + text_offset[1]
        ax.text(
            mx, my, text,
            ha="center", va="center",
            fontsize=7.5, fontweight="bold",
            color=color, family="Segoe UI",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor=color, alpha=0.92, lw=1.0),
            zorder=6
        )


def main():
    fig, ax = plt.subplots(figsize=(20, 24), dpi=300)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # =========================================================================
    # HEADER BANNER
    # =========================================================================
    header_box = FancyBboxPatch(
        (3, 93.5), 94, 5.2,
        boxstyle="round,pad=0.02,rounding_size=0.015",
        facecolor="#1E293B",
        edgecolor="#0F172A",
        linewidth=2.0,
        zorder=3
    )
    ax.add_patch(header_box)
    ax.text(
        50, 96.7, "HỆ THỐNG LIFEMATE VOICE-DRIVEN AI AGENT (ZALO AI CHALLENGE 2026)",
        ha="center", va="center", fontsize=15.5, fontweight="bold", color="#F8FAFC", family="Segoe UI", zorder=4
    )
    ax.text(
        50, 94.8, "SƠ ĐỒ HỢP NHẤT TOÀN DIỆN: HUẤN LUYỆN SFT + XỬ LÝ ÂM THANH + SUY LUẬN TỰ TRỊ + BỘ 3 LỚP PHÒNG VỆ + NỘP BÀI",
        ha="center", va="center", fontsize=9.2, color="#94A3B8", family="Segoe UI", zorder=4
    )

    # =========================================================================
    # STAGE 1: HUẤN LUYỆN OFFLINE SFT / QLORA (TOP LEFT & TOP MID)
    # =========================================================================
    stage1_bg = FancyBboxPatch((3, 76.5), 94, 15.5, boxstyle="round,pad=0.02,rounding_size=0.015",
                               facecolor="#EEF2FF", edgecolor="#6366F1", linewidth=2.0, linestyle="--", zorder=2)
    ax.add_patch(stage1_bg)
    ax.text(5.5, 90.3, "1. GIAI ĐOẠN HUẤN LUYỆN OFFLINE (SFT / QLORA TRAINING PIPELINE)",
            fontsize=11.5, fontweight="bold", color="#4338CA", family="Segoe UI", zorder=4)

    draw_box(ax, 5.5, 78.5, 18, 9.5, "train.jsonl (1.545 Mẫu)", "Dữ liệu kịch bản gốc từ BTC\n(Chứa audio, context, gold calls)", "#FFFFFF", "#818CF8")
    draw_box(ax, 27.5, 78.5, 20, 9.5, "scripts/prepare_sft_data.py", "Mô phỏng Teacher-Forcing từng bước\nSinh 5.086 ChatML Training Turns", "#FFFFFF", "#818CF8")
    draw_box(ax, 51.5, 78.5, 20, 9.5, "scripts/train_lora.py", "QLoRA 4-bit (NF4) + SDPA Attention\nr=16, alpha=32, batch=2, accum=4", "#FFFFFF", "#818CF8")
    draw_box(ax, 75.5, 78.5, 19.5, 9.5, "LoRA Adapter (~50MB)", "Lưu trên Google Drive\n/LifeMateAgent/adapters/qwen3_4b_lora", "#6366F1", "#4338CA", title_color="#FFFFFF", subtitle_color="#E0E7FF")

    draw_arrow(ax, (23.5, 83.2), (27.5, 83.2), "Bóc tách", "#6366F1")
    draw_arrow(ax, (47.5, 83.2), (51.5, 83.2), "5.086 turns", "#6366F1")
    draw_arrow(ax, (71.5, 83.2), (75.5, 83.2), "25-30 phút T4", "#6366F1")

    # =========================================================================
    # STAGE 2: TIẾP NHẬN DỮ LIỆU & PART B ASR (LEFT COLUMN)
    # =========================================================================
    stage2_bg = FancyBboxPatch((3, 44), 31, 30.5, boxstyle="round,pad=0.02,rounding_size=0.015",
                               facecolor="#F0F9FF", edgecolor="#0284C7", linewidth=2.0, zorder=2)
    ax.add_patch(stage2_bg)
    ax.text(5.5, 72.8, "2. TIẾP NHẬN DỮ LIỆU & PART B (ASR)", fontsize=11, fontweight="bold", color="#0369A1", family="Segoe UI", zorder=4)

    draw_box(ax, 5.5, 63, 26, 7.5, "Đầu Vào (Audio + Context)", "Audio (.wav / .mp3) + Context (current_time)\nLịch sử (History) + Tools Catalog (46 apps)", "#FFFFFF", "#38BDF8")
    draw_box(ax, 5.5, 53.5, 12, 7.2, "Whisper Turbo", "large-v3-turbo (800M)\nVAD + Prompt Bias", "#FFFFFF", "#38BDF8")
    draw_box(ax, 19.5, 53.5, 12, 7.2, "Precomputed Cache", "outputs/transcripts_*.json\nTiết kiệm 100% VRAM", "#FFFFFF", "#38BDF8")
    draw_box(ax, 5.5, 45.5, 26, 6.2, "Text Normalizer (normalizer.py)", "Bù số 0 ĐT, chuẩn hóa ngày giờ, bỏ nhiễu, NFC", "#0284C7", "#0369A1", title_color="#FFFFFF", subtitle_color="#E0F2FE")

    draw_arrow(ax, (11.5, 63), (11.5, 60.7), "", "#0284C7")
    draw_arrow(ax, (25.5, 63), (25.5, 60.7), "", "#0284C7")
    draw_arrow(ax, (11.5, 53.5), (15.5, 51.7), "", "#0284C7")
    draw_arrow(ax, (25.5, 53.5), (21.5, 51.7), "", "#0284C7")

    # =========================================================================
    # STAGE 3: PART C — NÃO BỘ AGENT CORE (CENTER COLUMN)
    # =========================================================================
    stage3_bg = FancyBboxPatch((36.5, 44), 31, 30.5, boxstyle="round,pad=0.02,rounding_size=0.015",
                               facecolor="#ECFDF5", edgecolor="#059669", linewidth=2.0, zorder=2)
    ax.add_patch(stage3_bg)
    ax.text(38.5, 72.8, "3. PART C — NÃO BỘ AGENT CORE (QWEN3-4B)", fontsize=11, fontweight="bold", color="#065F46", family="Segoe UI", zorder=4)

    draw_box(ax, 38.5, 64.5, 27, 6.5, "Tool Filter & Compact Schema", "Lọc công cụ cài đặt, nén signature\nGiảm 55% độ dài Prompt (10.8K -> 4.5K chars)", "#FFFFFF", "#34D399")
    draw_box(ax, 38.5, 55.5, 27, 7.2, "Prompt Builder & System Prompt", "Nghiêm cấm hỏi vặt, buộc tìm kiếm trước\nGhép Context + Tools + History + Observations", "#FFFFFF", "#34D399")
    draw_box(ax, 38.5, 45.5, 27, 8.2, "Qwen/Qwen3-4B-Instruct-2507", "Chạy SDPA Attention + Quản lý VRAM sạch sẽ\n+ Nạp LoRA Adapter (--adapter-path)", "#059669", "#047857", title_color="#FFFFFF", subtitle_color="#D1FAE5")

    # Connect stage 2 to stage 3
    draw_arrow(ax, (31.5, 48.6), (38.5, 58.5), "Transcript sạch", "#0284C7")
    draw_arrow(ax, (52, 64.5), (52, 62.7), "", "#059669")
    draw_arrow(ax, (52, 55.5), (52, 53.7), "", "#059669")
    
    # Connect LoRA from Stage 1 to Agent Core
    draw_arrow(ax, (85.2, 78.5), (65.5, 50), "Nạp Trọng Số LoRA", "#6366F1", rad=0.25, width=2.2, linestyle="--")

    # =========================================================================
    # STAGE 4: BỘ 3 LỚP PHÒNG VỆ AN TOÀN (RIGHT COLUMN)
    # =========================================================================
    stage4_bg = FancyBboxPatch((70, 44), 27, 30.5, boxstyle="round,pad=0.02,rounding_size=0.015",
                               facecolor="#FFFBEB", edgecolor="#D97706", linewidth=2.0, zorder=2)
    ax.add_patch(stage4_bg)
    ax.text(71.5, 72.8, "4. BỘ 3 CHỐT CHẶN PHÒNG VỆ (GUARDRAILS)", fontsize=10.5, fontweight="bold", color="#92400E", family="Segoe UI", zorder=4)

    draw_box(ax, 71.5, 64.5, 24, 6.8, "Response Parser", "Bóc tách: CALL: {tool, args} hoặc DONE\nHỗ trợ hàm lồng, JSON trần, đa định dạng", "#FFFFFF", "#FBBF24")
    draw_box(ax, 71.5, 54.5, 24, 8.2, "Task Tracker (Chống Lặp)", "CHẶN LẶP LỆNH GHI: write_count >= 1 -> STOP!\nBắt chu kỳ luân phiên (A-B-A-B) -> STOP!\n(Khắc phục triệt để 39% lỗi duplicate)", "#FFFFFF", "#FBBF24")
    draw_box(ax, 71.5, 45.2, 24, 7.5, "Hallucination Interceptor", "Kiểm tra ID tham số với Evidence Ledger\nNếu ID ảo -> Chặn lại và ép tìm kiếm trước\n(Khắc phục triệt để 16% lỗi failed calls)", "#D97706", "#B45309", title_color="#FFFFFF", subtitle_color="#FEF3C7")

    draw_arrow(ax, (65.5, 48), (71.5, 66.5), "Text sinh ra", "#059669")
    draw_arrow(ax, (83.5, 64.5), (83.5, 62.7), "Lệnh CALL", "#D97706")
    draw_arrow(ax, (83.5, 54.5), (83.5, 52.7), "Không lặp", "#D97706")

    # Feedback loop when fake ID is intercepted
    draw_arrow(ax, (71.5, 48.5), (65.5, 58.5), "Lỗi: 'ID ảo, phải tìm kiếm trước!'", "#DC2626", rad=0.2, linestyle=":")

    # =========================================================================
    # STAGE 5: MÔI TRƯỜNG THỰC THI & EVIDENCE LEDGER (BOTTOM MIDDLE)
    # =========================================================================
    stage5_bg = FancyBboxPatch((3, 21), 94, 20.5, boxstyle="round,pad=0.02,rounding_size=0.015",
                               facecolor="#FDF2F8", edgecolor="#DB2777", linewidth=2.0, zorder=2)
    ax.add_patch(stage5_bg)
    ax.text(5.5, 39.8, "5. MÔI TRƯỜNG MÔ PHỎNG (TOOL ENV) & SỔ CÁI BẰNG CHỨNG (EVIDENCE LEDGER)", fontsize=11.5, fontweight="bold", color="#9D174D", family="Segoe UI", zorder=4)

    draw_box(ax, 5.5, 23.5, 27, 14.5, "Evidence Ledger (Sổ Cái)", "Lưu trữ thực thể quan sát được (device, contact, bill, chat...)\nPhân giải NFC, chống nhầm lẫn ID giữa các app\nQuản lý độ tươi dữ liệu: Vô hiệu hóa ID cũ khi có lệnh ghi\nCung cấp danh sách ID thật cho Interceptor", "#FFFFFF", "#F472B6")
    draw_box(ax, 35.5, 23.5, 30, 14.5, "zaic_tools.ToolEnv (Mô Phỏng 46 Apps)", "Thực thi lệnh gọi trên state.json độc lập của từng mẫu\nĐảm bảo logic ứng dụng (chuyển tiền, bật đèn, gửi tin, đặt xe...)\nTrả về Observation: {'ok': true, 'data': ...}\nKhông có tool call rác / fake ID nào lọt vào môi trường!", "#FFFFFF", "#F472B6")
    draw_box(ax, 68.5, 23.5, 26.5, 14.5, "Observation Engine & Phản Hồi", "Định dạng kết quả ngắn gọn (max 600 chars)\nBơm kết quả vào Prompt cho bước suy luận tiếp theo\nCập nhật trạng thái bộ nhớ cho lần lặp kế tiếp", "#DB2777", "#BE185D", title_color="#FFFFFF", subtitle_color="#FCE7F3")

    # Connect stage 4 to stage 5
    draw_arrow(ax, (83.5, 45.2), (55, 38), "Lệnh hợp lệ: env.call()", "#DB2777", rad=-0.15)
    draw_arrow(ax, (50.5, 23.5), (20, 23.5), "Cập nhật thực thể mới", "#DB2777", rad=-0.3)
    draw_arrow(ax, (65.5, 30.5), (68.5, 30.5), "Kết quả", "#DB2777")
    draw_arrow(ax, (81.5, 38), (52, 55.5), "Observation cho bước tiếp theo", "#DB2777", rad=-0.2)

    # =========================================================================
    # STAGE 6: CHECKPOINT, STATE HASH & NỘP BÀI (BOTTOM ROW)
    # =========================================================================
    stage6_bg = FancyBboxPatch((3, 2), 94, 16.5, boxstyle="round,pad=0.02,rounding_size=0.015",
                               facecolor="#F0FDFA", edgecolor="#0D9488", linewidth=2.0, zorder=2)
    ax.add_patch(stage6_bg)
    ax.text(5.5, 16.8, "6. ĐÁNH GIÁ, SHA-256 STATE HASH & CHECKPOINT DRIVE (SUBMISSION ENGINE)", fontsize=11.5, fontweight="bold", color="#115E59", family="Segoe UI", zorder=4)

    draw_box(ax, 5.5, 4.5, 20, 10.5, "SHA-256 State Hash", "env.get_state_hash()\nKhớp 100% với ground truth BTC\n(Quyết định ăn trọn điểm F1)", "#FFFFFF", "#2DD4BF")
    draw_box(ax, 28.5, 4.5, 21, 10.5, "Submission Linter 7 Lớp", "Kiểm tra schema JSONL, ID mẫu\nTham số dict, hash 64 ký tự\n100% Structural Compliance", "#FFFFFF", "#2DD4BF")
    draw_box(ax, 52.5, 4.5, 21, 10.5, "Streaming & Auto-Resume", "outputs/submission_public_test.jsonl\nGhi đĩa sau MỖI mẫu (fout.flush)\nPhục hồi chính xác khi đứt mạng", "#FFFFFF", "#2DD4BF")
    draw_box(ax, 76.5, 4.5, 18.5, 10.5, "Google Drive Backup", "Đồng bộ mỗi 25 mẫu\n/LifeMateAgent/submissions\nSẵn sàng Nộp Bài 75 - 85+ Điểm!", "#0D9488", "#115E59", title_color="#FFFFFF", subtitle_color="#CCFBF1")

    # Connect to Stage 6
    draw_arrow(ax, (71.5, 68), (15.5, 15), "Lệnh DONE / Tracker Force DONE", "#0D9488", rad=0.25, width=2.2)
    draw_arrow(ax, (25.5, 9.7), (28.5, 9.7), "Xác thực", "#0D9488")
    draw_arrow(ax, (49.5, 9.7), (52.5, 9.7), "Ghi đĩa", "#0D9488")
    draw_arrow(ax, (73.5, 9.7), (76.5, 9.7), "Auto-Sync", "#0D9488")

    # Output paths
    output_path = Path("system_architecture.png").resolve()
    brain_path = Path(r"C:\Users\ASUS\.gemini\antigravity\brain\8b2174e6-e6ef-4a64-a471-18ce625b0ef2\system_architecture.png")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.savefig(brain_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"✅ Sơ đồ PNG 300 DPI đã xuất thành công tại: {output_path}")
    print(f"✅ Sơ đồ PNG đã đồng bộ vào Artifact: {brain_path}")


if __name__ == "__main__":
    main()
