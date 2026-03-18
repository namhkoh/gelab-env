# page_id: page_seatgeek_2ab99c22f31743719b11cf70dc6cb197_06
# screenshot: 2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-9.png
# step_index: 6/6
# task: Open SeatGeek. Search "Oracle Arena". Add the venue to the watch list.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for 1440x2960 canvas
w, h = canvas.size

# Clear background to pure white (match app background)
draw.rectangle([(0, 0), (w, h)], fill="#FFFFFF")

# Status bar area (top ~96px) - dark to match screenshot status bar
status_h = 96
draw.rectangle([(0, 0), (w, status_h)], fill="#000000")

# Hero/cover area (large dark header image background)
hero_bottom = 420
draw.rectangle([(0, 0), (w, hero_bottom)], fill="#000000")

# Subtle divider between hero and content
divider_y = hero_bottom
draw.line([(24, divider_y), (w - 24, divider_y)], fill="#E9E9E9", width=2)

# Main content area background (white) - starts directly under hero
content_top = divider_y
draw.rectangle([(0, content_top), (w, h)], fill="#FFFFFF")

# A top section card (rounded) that sits under the hero and will hold the venue title
card_margin_x = 30
card_top = 760
card_bottom = 1040
card_radius = 24
draw.rounded_rectangle(
    [(card_margin_x, card_top), (w - card_margin_x, card_bottom)],
    radius=card_radius,
    fill="#FFFFFF",
    outline="#EDEDED",
    width=1
)

# Separator lines for visual structure
# Thin line under the card to separate sections
sep_y1 = card_bottom + 8
draw.line([(24, sep_y1), (w - 24, sep_y1)], fill="#F3F3F3", width=1)

# Another faint divider further down the page for the "empty state" region
sep_y2 = 1600
draw.line([(24, sep_y2), (w - 24, sep_y2)], fill="#F3F3F3", width=1)

# Add subtle bottom fade area to give depth (very light grey rectangle)
bottom_fade_top = h - 220
draw.rectangle([(0, bottom_fade_top), (w, h)], fill="#FCFCFC")

# Left content margin guide (non-intrusive, very faint) for layout alignment
draw.line([(72, content_top + 16), (72, h - 16)], fill="#FFFFFF", width=1)  # virtually invisible, preserves structure

# Done. Icons and text will be pasted on top of these structural elements.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_06_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-9/00_icon_Share_this_performer.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1260, 84), _c0)
except Exception:
    pass
layout["Share_this_performer"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_06_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-9/01_icon_Track_this_performer.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1104, 84), _c1)
except Exception:
    pass
layout["Track_this_performer"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_06_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-9/02_icon_8.30_Wy.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (36, 84), _c2)
except Exception:
    pass
layout["8.30_Wy"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_06_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-9/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 59, 53)
    canvas.paste(_c3, (243, 9), _c3)
except Exception:
    pass
layout["icon_3"] = [243, 9, 302, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_06_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-9/04_icon_8.30_Wy.png
try:
    _c4 = get_crop(4, 53, 55)
    canvas.paste(_c4, (182, 8), _c4)
except Exception:
    pass
layout["8.30_Wy"] = [182, 8, 235, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_06_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-9/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 55, 52)
    canvas.paste(_c5, (313, 9), _c5)
except Exception:
    pass
layout["icon_5"] = [313, 9, 368, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_06_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-9/06_icon_8.30_Wy.png
try:
    _c6 = get_crop(6, 54, 59)
    canvas.paste(_c6, (118, 6), _c6)
except Exception:
    pass
layout["8.30_Wy"] = [118, 6, 172, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_06_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-9/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 96, 60)
    canvas.paste(_c7, (1217, 6), _c7)
except Exception:
    pass
layout["icon_7"] = [1217, 6, 1313, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_06_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-9/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 49, 58)
    canvas.paste(_c8, (1153, 8), _c8)
except Exception:
    pass
layout["icon_8"] = [1153, 8, 1202, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_06_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-9/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 53, 62)
    canvas.paste(_c9, (1319, 5), _c9)
except Exception:
    pass
layout["icon_9"] = [1319, 5, 1372, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_06_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-9/10_text_Oracle_Arena.png
try:
    _c10 = get_crop(10, 388, 66)
    canvas.paste(_c10, (57, 859), _c10)
except Exception:
    pass
layout["Oracle_Arena"] = [57, 859, 445, 925]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_06_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-9/11_text_No_upcoming_Shows.png
try:
    _c11 = get_crop(11, 526, 72)
    canvas.paste(_c11, (458, 1515), _c11)
except Exception:
    pass
layout["No_upcoming_Shows"] = [458, 1515, 984, 1587]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_06_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-9/12_text_Track_Oracle_Arena_for_event_updates.png
try:
    _c12 = get_crop(12, 855, 72)
    canvas.paste(_c12, (292, 1608), _c12)
except Exception:
    pass
layout["Track_Oracle_Arena_for_ev"] = [292, 1608, 1147, 1680]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_06_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-9/13_text_1Zt.png
try:
    _c13 = get_crop(13, 154, 142)
    canvas.paste(_c13, (646, 1289), _c13)
except Exception:
    pass
layout["1Zt"] = [646, 1289, 800, 1431]
