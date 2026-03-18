# page_id: page_seatgeek_42f99a380cf348fea73bd58c0e4ec8db_03
# screenshot: 2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6.png
# step_index: 3/14
# task: Open SeatGeek and search for the broadway show "lion king" on March 22. I need 3 tickets at average price less than 500 USD. Find the best seats and record the total price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the canvas (1440x2960)
# Uses provided variables: canvas (PIL Image) and draw (PIL ImageDraw)

# Base background
draw.rectangle((0, 0, 1440, 2960), fill="#ffffff")

# Status bar area (top ~50-90px)
status_h = 88
draw.rectangle((0, 0, 1440, status_h), fill="#f3f4f5")

# Subtle bottom edge for status bar
draw.line((0, status_h - 1, 1440, status_h - 1), fill="#e6e6e6", width=1)

# Search bar (rounded) - background only (do NOT draw icons/text inside)
search_left, search_top = 48, 120
search_right, search_bottom = 1392, 264  # height 144 as in detected elements
draw.rounded_rectangle(
    (search_left, search_top, search_right, search_bottom),
    radius=28,
    fill="#f7f8f9",
    outline="#e6e6e6",
    width=1
)

# Slight inner highlight for search bar (subtle)
draw.line((search_left + 8, search_bottom - 1, search_right - 8, search_bottom - 1), fill="#f0f0f0", width=1)

# Divider below search area (subtle)
divider_y1 = search_bottom + 48
draw.line((48, divider_y1, 1392, divider_y1), fill="#e9e9e9", width=1)

# Large thin section divider separating "Recent searches" block from suggestions
section_div_y = 1320
draw.line((48, section_div_y, 1392, section_div_y), fill="#e9e9e9", width=1)

# Suggestions card background (soft off-white block to group suggestion items)
suggest_card_top = section_div_y + 40
suggest_card_bottom = suggest_card_top + 520
draw.rounded_rectangle(
    (36, suggest_card_top, 1404, suggest_card_bottom),
    radius=12,
    fill="#fbfbfc",
    outline=None
)

# Add a faint inner separator line near top of suggestions card for subtle depth
draw.line((48, suggest_card_top + 28, 1392, suggest_card_top + 28), fill="#f2f2f2", width=1)

# Bottom navigation bar area (leave icons to be pasted on top)
nav_top = 2792
draw.rectangle((0, nav_top, 1440, 2960), fill="#ffffff")
# Top border for nav
draw.line((0, nav_top, 1440, nav_top), fill="#e9e9e9", width=1)
# Very subtle shadow above nav for depth
draw.line((0, nav_top + 1, 1440, nav_top + 1), fill="#f6f6f6", width=1)

# Subtle vertical padding guides (non-visible in final UI but help align structure)
# (Drawn very faint so they won't conflict with detected elements)
guide_color = "#fbfbfb"
draw.line((48, 0, 48, 2960), fill=guide_color, width=1)
draw.line((1392, 0, 1392, 2960), fill=guide_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 49, 69)
    canvas.paste(_c0, (1152, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1152, 0, 1201, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/01_icon_Tracking.png
try:
    _c1 = get_crop(1, 288, 168)
    canvas.paste(_c1, (864, 2792), _c1)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/02_icon_Brooklyn_Nets.png
try:
    _c2 = get_crop(2, 1440, 168)
    canvas.paste(_c2, (0, 639), _c2)
except Exception:
    pass
layout["Brooklyn_Nets"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 97, 67)
    canvas.paste(_c3, (1216, 0), _c3)
except Exception:
    pass
layout["icon_3"] = [1216, 0, 1313, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/04_icon_7.40_W.png
try:
    _c4 = get_crop(4, 168, 144)
    canvas.paste(_c4, (48, 120), _c4)
except Exception:
    pass
layout["7.40_W"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/05_icon_Browse.png
try:
    _c5 = get_crop(5, 288, 168)
    canvas.paste(_c5, (0, 2792), _c5)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/06_icon_Clear.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1248, 120), _c6)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/07_icon_Just_Announced_by_My_Performers.png
try:
    _c7 = get_crop(7, 1440, 168)
    canvas.paste(_c7, (0, 1688), _c7)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/08_icon_Tickets.png
try:
    _c8 = get_crop(8, 288, 168)
    canvas.paste(_c8, (576, 2792), _c8)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 53, 67)
    canvas.paste(_c9, (1319, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1319, 0, 1372, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/10_icon_Drake.png
try:
    _c10 = get_crop(10, 1440, 168)
    canvas.paste(_c10, (0, 807), _c10)
except Exception:
    pass
layout["Drake"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/11_icon_Recent_searches.png
try:
    _c11 = get_crop(11, 1440, 168)
    canvas.paste(_c11, (0, 471), _c11)
except Exception:
    pass
layout["Recent_searches"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/12_icon_Account.png
try:
    _c12 = get_crop(12, 288, 168)
    canvas.paste(_c12, (1152, 2792), _c12)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/13_icon_NBA_Playoffs.png
try:
    _c13 = get_crop(13, 1440, 168)
    canvas.paste(_c13, (0, 975), _c13)
except Exception:
    pass
layout["NBA_Playoffs"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/14_icon_Events_by_My_Performers.png
try:
    _c14 = get_crop(14, 1440, 168)
    canvas.paste(_c14, (0, 1520), _c14)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/15_icon_NBA_Playoffs.png
try:
    _c15 = get_crop(15, 1440, 168)
    canvas.paste(_c15, (0, 807), _c15)
except Exception:
    pass
layout["NBA_Playoffs"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/16_icon_GK.png
try:
    _c16 = get_crop(16, 63, 60)
    canvas.paste(_c16, (307, 3), _c16)
except Exception:
    pass
layout["GK"] = [307, 3, 370, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/17_icon_7.40_W.png
try:
    _c17 = get_crop(17, 68, 65)
    canvas.paste(_c17, (170, 2), _c17)
except Exception:
    pass
layout["7.40_W"] = [170, 2, 238, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/18_icon_Search.png
try:
    _c18 = get_crop(18, 288, 162)
    canvas.paste(_c18, (288, 2792), _c18)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/19_icon_Brooklyn_Nets.png
try:
    _c19 = get_crop(19, 1440, 168)
    canvas.paste(_c19, (0, 471), _c19)
except Exception:
    pass
layout["Brooklyn_Nets"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/20_icon_Drake.png
try:
    _c20 = get_crop(20, 1440, 168)
    canvas.paste(_c20, (0, 975), _c20)
except Exception:
    pass
layout["Drake"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/21_icon_Brooklyn_Nets.png
try:
    _c21 = get_crop(21, 1440, 168)
    canvas.paste(_c21, (0, 639), _c21)
except Exception:
    pass
layout["Brooklyn_Nets"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/22_icon_Search.png
try:
    _c22 = get_crop(22, 288, 162)
    canvas.paste(_c22, (288, 2792), _c22)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/23_icon_Performer_event_or_venue.png
try:
    _c23 = get_crop(23, 1032, 144)
    canvas.paste(_c23, (216, 120), _c23)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/24_icon_Just_Announced_by_My_Performers.png
try:
    _c24 = get_crop(24, 1440, 168)
    canvas.paste(_c24, (0, 1856), _c24)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/25_icon_Sofia_Isella.png
try:
    _c25 = get_crop(25, 1440, 168)
    canvas.paste(_c25, (0, 1143), _c25)
except Exception:
    pass
layout["Sofia_Isella"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/26_text_Recent_searches.png
try:
    _c26 = get_crop(26, 168, 144)
    canvas.paste(_c26, (48, 120), _c26)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/27_text_Sofia_Isella.png
try:
    _c27 = get_crop(27, 241, 49)
    canvas.paste(_c27, (234, 1203), _c27)
except Exception:
    pass
layout["Sofia_Isella"] = [234, 1203, 475, 1252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_03_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-6/28_text_Suggestions.png
try:
    _c28 = get_crop(28, 331, 74)
    canvas.paste(_c28, (40, 1423), _c28)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
