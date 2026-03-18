# page_id: page_seatgeek_2ab99c22f31743719b11cf70dc6cb197_02
# screenshot: 2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5.png
# step_index: 2/6
# task: Open SeatGeek. Search "Oracle Arena". Add the venue to the watch list.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the UI page using provided canvas and draw objects.

# Colors
bg_color = (250, 250, 250)        # very light grey background
status_color = (233, 233, 233)    # light grey status bar
card_color = (255, 255, 255)      # white card background
divider_color = (230, 230, 230)   # subtle divider
soft_shadow = (240, 240, 240)     # soft line to suggest separation

W, H = canvas.size

# Fill overall background
draw.rectangle([(0, 0), (W, H)], fill=bg_color)

# Status bar (top)
status_h = 96
draw.rectangle([(0, 0), (W, status_h)], fill=status_color)
# subtle bottom hairline under status bar
draw.line([(0, status_h), (W, status_h)], fill=soft_shadow, width=1)

# Main white rounded card that holds search + lists (inset from edges)
card_margin_x = 32
card_top = status_h + 8  # leave a small gap under status bar
card_bottom = 2720       # leave space for bottom nav
card_rect = [card_margin_x, card_top, W - card_margin_x, card_bottom]
draw.rounded_rectangle(card_rect, radius=28, fill=card_color, outline=divider_color, width=1)

# Divider under search area (approximate position)
# Note: the actual search field will be pasted on top; this is only the divider line
divider1_y = card_top + 180
draw.line([(card_margin_x + 32, divider1_y), (W - card_margin_x - 32, divider1_y)], fill=divider_color, width=2)

# Divider between Recent searches and Suggestions area
divider2_y = card_top + 920
draw.line([(card_margin_x + 32, divider2_y), (W - card_margin_x - 32, divider2_y)], fill=divider_color, width=2)

# Light section separators for list rows (suggest subtle row boundaries)
# These are faint and spaced to not conflict with pasted icons/text positions.
row_start_x = card_margin_x + 32
row_end_x = W - card_margin_x - 32
for y in range(int(divider1_y + 80), int(divider2_y - 60), 120):
    draw.line([(row_start_x, y), (row_end_x, y)], fill=(245, 245, 245), width=1)

# Subtle top shadow on the white card to lift it from the page
shadow_y = card_top + 2
draw.line([(card_margin_x + 8, shadow_y), (W - card_margin_x - 8, shadow_y)], fill=(245, 245, 245), width=1)

# Bottom navigation area separator (above nav icons)
nav_top = 2792
draw.line([(0, nav_top), (W, nav_top)], fill=divider_color, width=1)
# Slight fade area above nav to separate content from navigation (soft rectangle)
draw.rectangle([(0, nav_top - 8), (W, nav_top)], fill=(252, 252, 252))

# Small left and right inner vertical guides (visual structure hints, very faint)
guide_color = (248, 248, 248)
draw.line([(card_margin_x + 28, card_top + 8), (card_margin_x + 28, card_bottom - 8)], fill=guide_color, width=1)
draw.line([(W - card_margin_x - 28, card_top + 8), (W - card_margin_x - 28, card_bottom - 8)], fill=guide_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/00_icon_Shin_Lim.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 471), _c0)
except Exception:
    pass
layout["Shin_Lim"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/01_icon_WWE.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 639), _c1)
except Exception:
    pass
layout["WWE"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/02_icon_WWE.png
try:
    _c2 = get_crop(2, 1440, 168)
    canvas.paste(_c2, (0, 807), _c2)
except Exception:
    pass
layout["WWE"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 49, 69)
    canvas.paste(_c3, (1152, 0), _c3)
except Exception:
    pass
layout["icon_3"] = [1152, 0, 1201, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/04_icon_Tracking.png
try:
    _c4 = get_crop(4, 288, 168)
    canvas.paste(_c4, (864, 2792), _c4)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 64, 65)
    canvas.paste(_c5, (242, 2), _c5)
except Exception:
    pass
layout["icon_5"] = [242, 2, 306, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/06_icon_8.30_my.png
try:
    _c6 = get_crop(6, 168, 144)
    canvas.paste(_c6, (48, 120), _c6)
except Exception:
    pass
layout["8.30_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 98, 69)
    canvas.paste(_c7, (1215, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1215, 0, 1313, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/08_icon_Browse.png
try:
    _c8 = get_crop(8, 288, 168)
    canvas.paste(_c8, (0, 2792), _c8)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/09_icon_Tickets.png
try:
    _c9 = get_crop(9, 288, 168)
    canvas.paste(_c9, (576, 2792), _c9)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/10_icon_Madison_Square_Garden.png
try:
    _c10 = get_crop(10, 1440, 168)
    canvas.paste(_c10, (0, 975), _c10)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/11_icon_Just_Announced_by_My_Performers.png
try:
    _c11 = get_crop(11, 1440, 168)
    canvas.paste(_c11, (0, 1688), _c11)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/12_icon_The_Fonda_Theatre.png
try:
    _c12 = get_crop(12, 1440, 168)
    canvas.paste(_c12, (0, 807), _c12)
except Exception:
    pass
layout["The_Fonda_Theatre"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/13_icon_Clear.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 120), _c13)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 52, 68)
    canvas.paste(_c14, (1319, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1319, 0, 1371, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/15_icon_8.30_my.png
try:
    _c15 = get_crop(15, 47, 64)
    canvas.paste(_c15, (186, 1), _c15)
except Exception:
    pass
layout["8.30_my"] = [186, 1, 233, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/16_icon_Events_by_My_Performers.png
try:
    _c16 = get_crop(16, 1440, 168)
    canvas.paste(_c16, (0, 1520), _c16)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/17_icon_Dallas_Mavericks.png
try:
    _c17 = get_crop(17, 1440, 168)
    canvas.paste(_c17, (0, 639), _c17)
except Exception:
    pass
layout["Dallas_Mavericks"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/18_icon_Account.png
try:
    _c18 = get_crop(18, 288, 168)
    canvas.paste(_c18, (1152, 2792), _c18)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 62, 64)
    canvas.paste(_c19, (313, 2), _c19)
except Exception:
    pass
layout["icon_19"] = [313, 2, 375, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/20_icon_Dallas_Mavericks.png
try:
    _c20 = get_crop(20, 1440, 168)
    canvas.paste(_c20, (0, 471), _c20)
except Exception:
    pass
layout["Dallas_Mavericks"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/21_icon_Madison_Square_Garden.png
try:
    _c21 = get_crop(21, 1440, 168)
    canvas.paste(_c21, (0, 1143), _c21)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/22_icon_8.30_my.png
try:
    _c22 = get_crop(22, 58, 65)
    canvas.paste(_c22, (113, 0), _c22)
except Exception:
    pass
layout["8.30_my"] = [113, 0, 171, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/23_icon_Search.png
try:
    _c23 = get_crop(23, 288, 162)
    canvas.paste(_c23, (288, 2792), _c23)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/24_icon_Just_Announced_by_My_Performers.png
try:
    _c24 = get_crop(24, 1440, 168)
    canvas.paste(_c24, (0, 1856), _c24)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/25_icon_Performer_event_or_venue.png
try:
    _c25 = get_crop(25, 1032, 144)
    canvas.paste(_c25, (216, 120), _c25)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/26_text_Recent_searches.png
try:
    _c26 = get_crop(26, 168, 144)
    canvas.paste(_c26, (48, 120), _c26)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_02_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-5/27_text_Suggestions.png
try:
    _c27 = get_crop(27, 331, 74)
    canvas.paste(_c27, (40, 1423), _c27)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
