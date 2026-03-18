# page_id: page_seatgeek_2ab99c22f31743719b11cf70dc6cb197_05
# screenshot: 2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-8.png
# step_index: 5/6
# task: Open SeatGeek. Search "Oracle Arena". Add the venue to the watch list.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas (PIL Image) and draw (ImageDraw) are provided.
W, H = canvas.size

# Colors
hero_color = (10, 10, 10)        # deep black for hero image area
status_color = (6, 6, 6)         # slightly different black for status bar
divider_color = (230, 230, 230)  # light grey divider
card_shadow = (240, 240, 240)    # very light shadow
card_fill = (252, 252, 252)      # almost white card background
card_border = (236, 236, 236)    # card border

# Status bar and top hero band
status_h = 96
hero_h = int(H * 0.22)  # approximate hero banner height
if hero_h < 520:
    hero_h = 520

# Draw status bar background (top)
draw.rectangle([(0, 0), (W, status_h)], fill=status_color)

# Draw hero/banner area below status
draw.rectangle([(0, 0), (W, hero_h)], fill=hero_color)

# Divider line between hero and content
draw.line([(0, hero_h), (W, hero_h)], fill=divider_color, width=2)

# Content area (keeps white background) - no need to fill since canvas starts white.
# But draw a subtle full-width top content band to anchor the title area
content_band_h = 120
draw.rectangle([(0, hero_h), (W, hero_h + content_band_h)], fill=(255, 255, 255))
draw.line([(40, hero_h + content_band_h), (W - 40, hero_h + content_band_h)], fill=divider_color, width=1)

# Center "empty state" card area (rounded rectangle background behind the group)
card_w = 760
card_h = 420
card_cx = W // 2
card_cy = hero_h + 900  # place it well below the hero area, leave space for title
card_left = card_cx - card_w // 2
card_top = card_cy - card_h // 2
card_right = card_left + card_w
card_bottom = card_top + card_h
card_radius = 28

# Shadow behind card (slightly bigger rounded rect)
shadow_inset = 8
draw.rounded_rectangle(
    [(card_left - shadow_inset, card_top + 8),
     (card_right + shadow_inset, card_bottom + 8)],
    radius=card_radius + 4,
    fill=card_shadow
)

# Card background and border
draw.rounded_rectangle(
    [(card_left, card_top), (card_right, card_bottom)],
    radius=card_radius,
    fill=card_fill,
    outline=card_border,
    width=1
)

# Additional subtle separators for structure:
# 1) A thin left-aligned vertical guide near title area (visual structure only)
guide_x = 40
draw.line([(guide_x, hero_h + 20), (guide_x, H - 200)], fill=(245, 245, 245), width=1)

# 2) A faint horizontal rule above the card to separate top content from the "empty state"
rule_y = card_top - 80
draw.line([(60, rule_y), (W - 60, rule_y)], fill=(245, 245, 245), width=1)

# 3) Footer thin divider far down the page to indicate end of content area
footer_div_y = card_bottom + 240
draw.line([(40, footer_div_y), (W - 40, footer_div_y)], fill=divider_color, width=1)

# Done: background, status bar, hero/banner, card background, and separators drawn.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_05_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-8/00_icon_Track_Now.png
try:
    _c0 = get_crop(0, 337, 153)
    canvas.paste(_c0, (551, 1638), _c0)
except Exception:
    pass
layout["Track_Now"] = [551, 1638, 888, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_05_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-8/01_icon_Track_this_performer.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1104, 84), _c1)
except Exception:
    pass
layout["Track_this_performer"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_05_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-8/02_icon_Share_this_performer.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1260, 84), _c2)
except Exception:
    pass
layout["Share_this_performer"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_05_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-8/03_icon_8.30_Wy.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (36, 84), _c3)
except Exception:
    pass
layout["8.30_Wy"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_05_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-8/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 58, 53)
    canvas.paste(_c4, (244, 9), _c4)
except Exception:
    pass
layout["icon_4"] = [244, 9, 302, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_05_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-8/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 55, 52)
    canvas.paste(_c5, (313, 9), _c5)
except Exception:
    pass
layout["icon_5"] = [313, 9, 368, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_05_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-8/06_icon_8.30_Wy.png
try:
    _c6 = get_crop(6, 53, 55)
    canvas.paste(_c6, (182, 8), _c6)
except Exception:
    pass
layout["8.30_Wy"] = [182, 8, 235, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_05_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-8/07_icon_8.30_Wy.png
try:
    _c7 = get_crop(7, 54, 58)
    canvas.paste(_c7, (118, 6), _c7)
except Exception:
    pass
layout["8.30_Wy"] = [118, 6, 172, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_05_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-8/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 48, 57)
    canvas.paste(_c8, (1154, 8), _c8)
except Exception:
    pass
layout["icon_8"] = [1154, 8, 1202, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_05_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-8/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 99, 62)
    canvas.paste(_c9, (1218, 5), _c9)
except Exception:
    pass
layout["icon_9"] = [1218, 5, 1317, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_05_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-8/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 46, 56)
    canvas.paste(_c10, (1327, 6), _c10)
except Exception:
    pass
layout["icon_10"] = [1327, 6, 1373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_05_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-8/11_text_Oracle_Arena.png
try:
    _c11 = get_crop(11, 388, 66)
    canvas.paste(_c11, (57, 859), _c11)
except Exception:
    pass
layout["Oracle_Arena"] = [57, 859, 445, 925]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_05_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-8/12_text_No_upcoming_Shows.png
try:
    _c12 = get_crop(12, 337, 153)
    canvas.paste(_c12, (551, 1638), _c12)
except Exception:
    pass
layout["No_upcoming_Shows"] = [551, 1638, 888, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_05_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-8/13_text_Track_Oracle_Arena_for_event_updates.png
try:
    _c13 = get_crop(13, 337, 153)
    canvas.paste(_c13, (551, 1638), _c13)
except Exception:
    pass
layout["Track_Oracle_Arena_for_ev"] = [551, 1638, 888, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_05_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-8/14_text_1Zz.png
try:
    _c14 = get_crop(14, 156, 142)
    canvas.paste(_c14, (645, 1176), _c14)
except Exception:
    pass
layout["1Zz"] = [645, 1176, 801, 1318]
