# page_id: page_seatgeek_2ab99c22f31743719b11cf70dc6cb197_03
# screenshot: 2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6.png
# step_index: 3/6
# task: Open SeatGeek. Search "Oracle Arena". Add the venue to the watch list.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the canvas provided.
# Available variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
status_bar_color = (242, 242, 242)    # light gray for status bar
search_bg = (250, 250, 250)           # very light gray for search box
search_border = (233, 233, 233)       # subtle border for search box
divider = (230, 230, 230)             # neutral divider lines
soft_divider = (240, 240, 240)        # even softer dividers
card_outline = (243, 243, 243)        # card outline
nav_border = (230, 230, 230)          # top border of bottom nav
shadow = (245, 245, 245)              # faint shadow/ground

W, H = canvas.size

# 1) Status bar area at top
status_h = 96
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# subtle bottom hairline for status bar
draw.line([(0, status_h - 1), (W, status_h - 1)], fill=divider, width=1)

# 2) Search bar (rounded) below status bar
search_x0, search_y0 = 24, 64
search_x1, search_y1 = W - 24, 200
search_radius = 32
draw.rounded_rectangle(
    [(search_x0, search_y0), (search_x1, search_y1)],
    radius=search_radius,
    fill=search_bg,
    outline=search_border,
    width=2
)

# subtle divider under the search area
draw.line([(24, search_y1 + 16), (W - 24, search_y1 + 16)], fill=soft_divider, width=1)

# 3) Content grouping card (Recent searches section) background
# Place a very light rounded card behind the list of recent searches (keeps icons/text free)
recent_card_x0, recent_card_y0 = 24, search_y1 + 32
recent_card_x1, recent_card_y1 = W - 24, 1344
recent_card_radius = 12

# faint shadow top (single light line) to lift the card slightly
draw.rounded_rectangle(
    [(recent_card_x0, recent_card_y0 + 2), (recent_card_x1, recent_card_y1 + 2)],
    radius=recent_card_radius,
    fill=shadow,
    outline=None
)

# card body (kept visually minimal — mostly white so it doesn't duplicate icons/text)
draw.rounded_rectangle(
    [(recent_card_x0, recent_card_y0), (recent_card_x1, recent_card_y1)],
    radius=recent_card_radius,
    fill=(255, 255, 255),
    outline=card_outline,
    width=1
)

# 4) Separator lines between list items inside the recent searches card
# Using detected list rows positions as guides: items stacked roughly every 168px starting near y ~471
# We'll draw separators at those bottoms so icons/text will be pasted on top later.
list_start_y = 471
row_height = 168
separators = []
# Draw enough separators to cover the visible list area
for i in range(1, 7):
    y = list_start_y + i * row_height
    if recent_card_y0 < y < recent_card_y1:
        separators.append(y)
        draw.line([(recent_card_x0 + 12, y), (recent_card_x1 - 12, y)], fill=soft_divider, width=1)

# 5) Major divider between recent searches block and Suggestions area
major_div_y = recent_card_y1 + 32
draw.line([(24, major_div_y), (W - 24, major_div_y)], fill=divider, width=2)

# 6) Suggestions header separator (a lighter divider a bit above the suggestions header)
suggestions_div_y = major_div_y + 88
draw.line([(24, suggestions_div_y), (W - 24, suggestions_div_y)], fill=soft_divider, width=1)

# 7) Bottom navigation bar area and top border
nav_top = 2792  # based on detected icon positions for nav
draw.rectangle([(0, nav_top), (W, H)], fill=(255, 255, 255))
# top border (separator) for nav bar
draw.line([(0, nav_top), (W, nav_top)], fill=nav_border, width=1)

# 8) Small horizontal left/right margins guideline (subtle) to match app structure edges
# (These are faint and only guide structure; they won't overlap detected icons)
draw.line([(24, 0), (24, H)], fill=(255, 255, 255), width=0)  # no-op visual placeholder
draw.line([(W - 24, 0), (W - 24, H)], fill=(255, 255, 255), width=0)

# End of background/structure drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/00_icon_WWE.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 807), _c0)
except Exception:
    pass
layout["WWE"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/01_icon_WWE.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 639), _c1)
except Exception:
    pass
layout["WWE"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/02_icon_8.30_my.png
try:
    _c2 = get_crop(2, 168, 144)
    canvas.paste(_c2, (48, 120), _c2)
except Exception:
    pass
layout["8.30_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 47, 70)
    canvas.paste(_c3, (1153, 0), _c3)
except Exception:
    pass
layout["icon_3"] = [1153, 0, 1200, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/04_icon_Tracking.png
try:
    _c4 = get_crop(4, 288, 168)
    canvas.paste(_c4, (864, 2792), _c4)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/05_icon_Recent_searches.png
try:
    _c5 = get_crop(5, 1440, 168)
    canvas.paste(_c5, (0, 471), _c5)
except Exception:
    pass
layout["Recent_searches"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/06_icon_Browse.png
try:
    _c6 = get_crop(6, 288, 168)
    canvas.paste(_c6, (0, 2792), _c6)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 61, 64)
    canvas.paste(_c7, (243, 2), _c7)
except Exception:
    pass
layout["icon_7"] = [243, 2, 304, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/08_icon_Just_Announced_by_My_Performers.png
try:
    _c8 = get_crop(8, 1440, 168)
    canvas.paste(_c8, (0, 1688), _c8)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/09_icon_Madison_Square_Garden.png
try:
    _c9 = get_crop(9, 1440, 168)
    canvas.paste(_c9, (0, 975), _c9)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/10_icon_Tickets.png
try:
    _c10 = get_crop(10, 288, 168)
    canvas.paste(_c10, (576, 2792), _c10)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/11_icon_The_Fonda_Theatre.png
try:
    _c11 = get_crop(11, 1440, 168)
    canvas.paste(_c11, (0, 807), _c11)
except Exception:
    pass
layout["The_Fonda_Theatre"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 95, 68)
    canvas.paste(_c12, (1217, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1217, 0, 1312, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/13_icon_Clear.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 120), _c13)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/14_icon_Dallas_Mavericks.png
try:
    _c14 = get_crop(14, 1440, 168)
    canvas.paste(_c14, (0, 471), _c14)
except Exception:
    pass
layout["Dallas_Mavericks"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/15_icon_Dallas_Mavericks.png
try:
    _c15 = get_crop(15, 1440, 168)
    canvas.paste(_c15, (0, 639), _c15)
except Exception:
    pass
layout["Dallas_Mavericks"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/16_icon_Events_by_My_Performers.png
try:
    _c16 = get_crop(16, 1440, 168)
    canvas.paste(_c16, (0, 1520), _c16)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 52, 68)
    canvas.paste(_c17, (1319, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1319, 0, 1371, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/18_icon_Account.png
try:
    _c18 = get_crop(18, 288, 168)
    canvas.paste(_c18, (1152, 2792), _c18)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 59, 64)
    canvas.paste(_c19, (313, 2), _c19)
except Exception:
    pass
layout["icon_19"] = [313, 2, 372, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/20_icon_Madison_Square_Garden.png
try:
    _c20 = get_crop(20, 1440, 168)
    canvas.paste(_c20, (0, 1143), _c20)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/21_icon_8.30_my.png
try:
    _c21 = get_crop(21, 46, 63)
    canvas.paste(_c21, (186, 1), _c21)
except Exception:
    pass
layout["8.30_my"] = [186, 1, 232, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/22_icon_Search.png
try:
    _c22 = get_crop(22, 288, 162)
    canvas.paste(_c22, (288, 2792), _c22)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/23_icon_Performer_event_or_venue.png
try:
    _c23 = get_crop(23, 1032, 144)
    canvas.paste(_c23, (216, 120), _c23)
except Exception:
    pass
layout["Performer;_event;_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/24_icon_Search.png
try:
    _c24 = get_crop(24, 288, 162)
    canvas.paste(_c24, (288, 2792), _c24)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/25_icon_Just_Announced_by_My_Performers.png
try:
    _c25 = get_crop(25, 1440, 168)
    canvas.paste(_c25, (0, 1856), _c25)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/26_text_8.30_my.png
try:
    _c26 = get_crop(26, 156, 52)
    canvas.paste(_c26, (16, 9), _c26)
except Exception:
    pass
layout["8.30_my"] = [16, 9, 172, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/27_text_Recent_searches.png
try:
    _c27 = get_crop(27, 168, 144)
    canvas.paste(_c27, (48, 120), _c27)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_03_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-6/28_text_Suggestions.png
try:
    _c28 = get_crop(28, 331, 74)
    canvas.paste(_c28, (40, 1423), _c28)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
