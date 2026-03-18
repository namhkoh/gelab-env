# page_id: page_seatgeek_3675f36a85734af4aa90c8115351dd12_02
# screenshot: 2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5.png
# step_index: 2/9
# task: Open SeatGeek. Search "The Fonda Theatre". Select the top popular event and track it. What is the lowest price?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint overall background
draw.rectangle([(0, 0), (canvas.width, canvas.height)], fill="#FBFBFB")

# Top status bar (approx 72px tall)
status_bar_h = 72
draw.rectangle([(0, 0), (canvas.width, status_bar_h)], fill="#ECECEC")

# Subtle divider under status bar to separate from header region
draw.line([(0, status_bar_h), (canvas.width, status_bar_h)], fill="#E6E6E6", width=1)

# Search box background (rounded) - leave icons/text to be pasted on top
search_left = 48
search_right = canvas.width - 48
search_top = 120
search_bottom = 264
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=28,
    fill="#F6F7F8",
    outline="#E5E5E5",
    width=1
)

# Thin divider line below the search box
divider_y = search_bottom + 24
draw.line([(search_left, divider_y), (search_right, divider_y)], fill="#E9E9E9", width=1)

# Draw separators between list rows (use left inset so icons + text area remain untouched)
list_left = 48
list_right = canvas.width - 48
# Row heights in this UI are roughly 168px blocks starting near y=471
row_tops = [639, 807, 975, 1143, 1520, 1688, 1856]
for y in row_tops:
    draw.line([(list_left, y), (list_right, y)], fill="#EFEFEF", width=1)

# Additional subtle section divider under "Recent searches" heading area
heading_divider_y = 420
draw.line([(list_left, heading_divider_y), (list_right, heading_divider_y)], fill="#EEEAEA", width=1)

# Light card-style background for the suggestions block (keeps contrast but no content)
suggestions_top = 1400
suggestions_bottom = 2000
card_margin = 36
draw.rounded_rectangle(
    [(card_margin, suggestions_top), (canvas.width - card_margin, suggestions_bottom)],
    radius=16,
    fill="#FFFFFF",
    outline=None
)

# Subtle horizontal separators inside suggestions card (inset)
suggestion_sep_y = [1520, 1688, 1856]
for y in suggestion_sep_y:
    draw.line([(card_margin + 12, y), (canvas.width - card_margin - 12, y)], fill="#F0F0F0", width=1)

# Bottom navigation bar background (approx 168px tall) with top border
nav_top = 2792
nav_bottom = canvas.height
draw.rectangle([(0, nav_top), (canvas.width, nav_bottom)], fill="#FFFFFF")
draw.line([(0, nav_top), (canvas.width, nav_top)], fill="#E7E7E7", width=1)

# Slight top shadow for the nav bar to lift it off the page (a faint thin band)
shadow_y = nav_top + 2
draw.line([(0, shadow_y), (canvas.width, shadow_y)], fill="#F6F6F6", width=1)

# Small visual accent: very faint vertical rules to subtly group content areas (non-intrusive)
accent_x_positions = [360, 720, 1080]
for x in accent_x_positions:
    draw.line([(x, search_bottom + 40), (x, nav_top - 40)], fill="#FBFBFB", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/00_icon_Ed_Sheeran.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 639), _c0)
except Exception:
    pass
layout["Ed_Sheeran"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 49, 69)
    canvas.paste(_c1, (1152, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1152, 0, 1201, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/02_icon_Tracking.png
try:
    _c2 = get_crop(2, 288, 168)
    canvas.paste(_c2, (864, 2792), _c2)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 64, 64)
    canvas.paste(_c3, (242, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [242, 2, 306, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/04_icon_8.11_Wy.png
try:
    _c4 = get_crop(4, 55, 65)
    canvas.paste(_c4, (114, 0), _c4)
except Exception:
    pass
layout["8.11_Wy"] = [114, 0, 169, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/05_icon_Madison_Square_Garden.png
try:
    _c5 = get_crop(5, 1440, 168)
    canvas.paste(_c5, (0, 471), _c5)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/06_icon_Browse.png
try:
    _c6 = get_crop(6, 288, 168)
    canvas.paste(_c6, (0, 2792), _c6)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 98, 69)
    canvas.paste(_c7, (1215, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1215, 0, 1313, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/08_icon_Just_Announced_by_My_Performers.png
try:
    _c8 = get_crop(8, 1440, 168)
    canvas.paste(_c8, (0, 1688), _c8)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/09_icon_Ed_Sheeran.png
try:
    _c9 = get_crop(9, 1440, 168)
    canvas.paste(_c9, (0, 807), _c9)
except Exception:
    pass
layout["Ed_Sheeran"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/10_icon_Tickets.png
try:
    _c10 = get_crop(10, 288, 168)
    canvas.paste(_c10, (576, 2792), _c10)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/11_icon_Metropolitan_Opera.png
try:
    _c11 = get_crop(11, 1440, 168)
    canvas.paste(_c11, (0, 807), _c11)
except Exception:
    pass
layout["Metropolitan_Opera"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/12_icon_8.11_Wy.png
try:
    _c12 = get_crop(12, 47, 64)
    canvas.paste(_c12, (186, 1), _c12)
except Exception:
    pass
layout["8.11_Wy"] = [186, 1, 233, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/13_icon_Metropolitan_Opera.png
try:
    _c13 = get_crop(13, 1440, 168)
    canvas.paste(_c13, (0, 975), _c13)
except Exception:
    pass
layout["Metropolitan_Opera"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/14_icon_Clear.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 120), _c14)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/15_icon_8.11_Wy.png
try:
    _c15 = get_crop(15, 168, 144)
    canvas.paste(_c15, (48, 120), _c15)
except Exception:
    pass
layout["8.11_Wy"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 52, 68)
    canvas.paste(_c16, (1319, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1319, 0, 1371, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/17_icon_Account.png
try:
    _c17 = get_crop(17, 288, 168)
    canvas.paste(_c17, (1152, 2792), _c17)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/18_icon_Madison_Square_Garden.png
try:
    _c18 = get_crop(18, 1440, 168)
    canvas.paste(_c18, (0, 639), _c18)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 61, 63)
    canvas.paste(_c19, (313, 2), _c19)
except Exception:
    pass
layout["icon_19"] = [313, 2, 374, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/20_icon_Events_by_My_Performers.png
try:
    _c20 = get_crop(20, 1440, 168)
    canvas.paste(_c20, (0, 1520), _c20)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/21_icon_Suggestions.png
try:
    _c21 = get_crop(21, 1440, 168)
    canvas.paste(_c21, (0, 1143), _c21)
except Exception:
    pass
layout["Suggestions"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/22_icon_Search.png
try:
    _c22 = get_crop(22, 288, 162)
    canvas.paste(_c22, (288, 2792), _c22)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/23_icon_Just_Announced_by_My_Performers.png
try:
    _c23 = get_crop(23, 1440, 168)
    canvas.paste(_c23, (0, 1856), _c23)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/24_icon_8.11_Wy.png
try:
    _c24 = get_crop(24, 91, 66)
    canvas.paste(_c24, (13, 1), _c24)
except Exception:
    pass
layout["8.11_Wy"] = [13, 1, 104, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/25_icon_Performer_event_or_venue.png
try:
    _c25 = get_crop(25, 1032, 144)
    canvas.paste(_c25, (216, 120), _c25)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/26_icon_Los_Angeles_Lakers.png
try:
    _c26 = get_crop(26, 1440, 168)
    canvas.paste(_c26, (0, 1143), _c26)
except Exception:
    pass
layout["Los_Angeles_Lakers"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/27_text_Recent_searches.png
try:
    _c27 = get_crop(27, 168, 144)
    canvas.paste(_c27, (48, 120), _c27)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_02_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-5/28_text_Suggestions.png
try:
    _c28 = get_crop(28, 331, 74)
    canvas.paste(_c28, (40, 1423), _c28)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
