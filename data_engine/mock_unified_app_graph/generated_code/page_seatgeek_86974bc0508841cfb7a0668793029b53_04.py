# page_id: page_seatgeek_86974bc0508841cfb7a0668793029b53_04
# screenshot: 2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7.png
# step_index: 4/5
# task: Open SeatGeek. Search for the "Ed Sheeran" concert. Check the next upcoming event. When and where is the concert?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background & structural UI elements for the mobile UI (1440x2960)
# Available variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Colors
bg_color = "#ffffff"
status_bg = "#f4f5f6"
search_bg = "#fbfbfb"
muted_div = "#ededed"
card_border = "#f1f1f1"
card_bg = "#ffffff"
bottom_nav_bg = "#ffffff"
shadow_line = "#efefef"

# Fill overall background (canvas starts white, but ensure consistent color)
draw.rectangle([(0, 0), (W, H)], fill=bg_color)

# Status bar area (top ~80px)
status_h = 80
draw.rectangle([(0, 0), (W, status_h)], fill=status_bg)

# Subtle divider under status/search area
draw.line([(24, status_h), (W - 24, status_h)], fill=muted_div, width=1)

# Search bar background (rounded rect)
search_x1, search_y1 = 32, 56
search_x2, search_y2 = W - 32, 156
search_radius = 24
draw.rounded_rectangle([search_x1, search_y1, search_x2, search_y2],
                       radius=search_radius, fill=search_bg, outline=card_border, width=1)

# Thin divider under the search area to separate from content
divider_y = 188
draw.line([(24, divider_y), (W - 24, divider_y)], fill=muted_div, width=1)

# Section group cards / subtle backgrounds
# Top Results group card (subtle elevated card)
top_group_y1, top_group_y2 = 320, 660
draw.rounded_rectangle([24, top_group_y1, W - 24, top_group_y2],
                       radius=12, fill=card_bg, outline=card_border, width=1)
# Light shadow line under top group
draw.line([(24, top_group_y2 + 1), (W - 24, top_group_y2 + 1)], fill=shadow_line, width=1)

# Performers group card
perf_group_y1, perf_group_y2 = 700, 1180
draw.rounded_rectangle([24, perf_group_y1, W - 24, perf_group_y2],
                       radius=12, fill=card_bg, outline=card_border, width=1)
draw.line([(24, perf_group_y2 + 1), (W - 24, perf_group_y2 + 1)], fill=shadow_line, width=1)

# Events group card (holds list of events)
events_group_y1, events_group_y2 = 1840, 2440
draw.rounded_rectangle([24, events_group_y1, W - 24, events_group_y2],
                       radius=12, fill=card_bg, outline=card_border, width=1)
draw.line([(24, events_group_y2 + 1), (W - 24, events_group_y2 + 1)], fill=shadow_line, width=1)

# Venues section area (below events, before bottom nav)
venues_y1, venues_y2 = 2490, 2740
draw.rounded_rectangle([24, venues_y1, W - 24, venues_y2],
                       radius=12, fill=card_bg, outline=card_border, width=1)

# Draw separators between list items / sections using detected-like y positions (thin subtle lines)
separator_positions = [650, 829, 1217, 1396, 1575, 1868, 2142, 2321, 2792]
for y in separator_positions:
    # Ensure separators are within canvas
    if 0 < y < H:
        draw.line([(24, y), (W - 24, y)], fill=muted_div, width=1)

# Header divider above bottom navigation
bottom_nav_top = 2792
draw.line([(0, bottom_nav_top), (W, bottom_nav_top)], fill=muted_div, width=1)

# Bottom navigation background
draw.rectangle([(0, bottom_nav_top), (W, H)], fill=bottom_nav_bg)
# Slight top shadow for bottom nav
draw.line([(0, bottom_nav_top + 1), (W, bottom_nav_top + 1)], fill=shadow_line, width=2)

# Small left/right safe area vertical guides (subtle, non-intrusive)
side_margin_x = 24
draw.line([(side_margin_x, status_h + 6), (side_margin_x, H - 120)], fill="#ffffff00")  # invisible guide
draw.line([(W - side_margin_x, status_h + 6), (W - side_margin_x, H - 120)], fill="#ffffff00")

# Completed structural drawing. (No text/icons drawn — those will be pasted on top.)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/00_icon_No_events.png
try:
    _c0 = get_crop(0, 1440, 179)
    canvas.paste(_c0, (0, 1575), _c0)
except Exception:
    pass
layout["No_events"] = [0, 1575, 1440, 1754]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/01_icon_Performers.png
try:
    _c1 = get_crop(1, 1440, 179)
    canvas.paste(_c1, (0, 1217), _c1)
except Exception:
    pass
layout["Performers"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/02_icon_No_events.png
try:
    _c2 = get_crop(2, 1440, 179)
    canvas.paste(_c2, (0, 1396), _c2)
except Exception:
    pass
layout["No_events"] = [0, 1396, 1440, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/03_icon_Top_results.png
try:
    _c3 = get_crop(3, 1440, 179)
    canvas.paste(_c3, (0, 471), _c3)
except Exception:
    pass
layout["Top_results"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/04_icon_No_events.png
try:
    _c4 = get_crop(4, 1440, 179)
    canvas.paste(_c4, (0, 650), _c4)
except Exception:
    pass
layout["No_events"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/05_icon_Fri.png
try:
    _c5 = get_crop(5, 1440, 179)
    canvas.paste(_c5, (0, 829), _c5)
except Exception:
    pass
layout["Fri,"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/06_icon_Fri.png
try:
    _c6 = get_crop(6, 1440, 179)
    canvas.paste(_c6, (0, 1963), _c6)
except Exception:
    pass
layout["Fri,"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 43, 70)
    canvas.paste(_c7, (1155, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1155, 0, 1198, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/08_icon_8.00_my.png
try:
    _c8 = get_crop(8, 168, 144)
    canvas.paste(_c8, (48, 120), _c8)
except Exception:
    pass
layout["8.00_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 59, 61)
    canvas.paste(_c9, (244, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [244, 3, 303, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 93, 69)
    canvas.paste(_c10, (1219, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1219, 0, 1312, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/11_icon_Hollywood_FL.png
try:
    _c11 = get_crop(11, 1440, 179)
    canvas.paste(_c11, (0, 1963), _c11)
except Exception:
    pass
layout["Hollywood,_FL"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/12_icon_Sun.png
try:
    _c12 = get_crop(12, 1440, 179)
    canvas.paste(_c12, (0, 2321), _c12)
except Exception:
    pass
layout["Sun,"] = [0, 2321, 1440, 2500]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 54, 60)
    canvas.paste(_c13, (315, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [315, 3, 369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/14_icon_8.00_my.png
try:
    _c14 = get_crop(14, 45, 62)
    canvas.paste(_c14, (187, 1), _c14)
except Exception:
    pass
layout["8.00_my"] = [187, 1, 232, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/15_icon_Fri.png
try:
    _c15 = get_crop(15, 1440, 179)
    canvas.paste(_c15, (0, 2142), _c15)
except Exception:
    pass
layout["Fri,"] = [0, 2142, 1440, 2321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 45, 65)
    canvas.paste(_c16, (1326, 2), _c16)
except Exception:
    pass
layout["icon_16"] = [1326, 2, 1371, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/17_icon_Napa_CA.png
try:
    _c17 = get_crop(17, 1440, 179)
    canvas.paste(_c17, (0, 2321), _c17)
except Exception:
    pass
layout["Napa,_CA"] = [0, 2321, 1440, 2500]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/18_icon_Napa_CA.png
try:
    _c18 = get_crop(18, 1440, 179)
    canvas.paste(_c18, (0, 2142), _c18)
except Exception:
    pass
layout["Napa,_CA"] = [0, 2142, 1440, 2321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/19_icon_Clear.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1248, 120), _c19)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/20_icon_Tickets.png
try:
    _c20 = get_crop(20, 288, 168)
    canvas.paste(_c20, (576, 2792), _c20)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/21_icon_Ed_Sheeran_Thursday.png
try:
    _c21 = get_crop(21, 1440, 179)
    canvas.paste(_c21, (0, 471), _c21)
except Exception:
    pass
layout["Ed_Sheeran_Thursday"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/22_icon_Ed_Sheeran.png
try:
    _c22 = get_crop(22, 249, 57)
    canvas.paste(_c22, (234, 860), _c22)
except Exception:
    pass
layout["Ed_Sheeran"] = [234, 860, 483, 917]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/23_icon_Sheraton_Grand_Hotel.png
try:
    _c23 = get_crop(23, 288, 162)
    canvas.paste(_c23, (288, 2792), _c23)
except Exception:
    pass
layout["Sheraton_Grand_Hotel"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/24_icon_Parking_for_Ed_Sheeran_Concert.png
try:
    _c24 = get_crop(24, 1440, 179)
    canvas.paste(_c24, (0, 1575), _c24)
except Exception:
    pass
layout["Parking_for_Ed_Sheeran_Co"] = [0, 1575, 1440, 1754]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/25_icon_8.00_my.png
try:
    _c25 = get_crop(25, 52, 62)
    canvas.paste(_c25, (117, 1), _c25)
except Exception:
    pass
layout["8.00_my"] = [117, 1, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/26_icon_Ed_Sheeran.png
try:
    _c26 = get_crop(26, 252, 59)
    canvas.paste(_c26, (234, 1248), _c26)
except Exception:
    pass
layout["Ed_Sheeran"] = [234, 1248, 486, 1307]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/27_icon_Hollywood_FL.png
try:
    _c27 = get_crop(27, 1440, 179)
    canvas.paste(_c27, (0, 650), _c27)
except Exception:
    pass
layout["Hollywood,_FL"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/28_icon_Hollywood_FL.png
try:
    _c28 = get_crop(28, 1440, 179)
    canvas.paste(_c28, (0, 829), _c28)
except Exception:
    pass
layout["Hollywood,_FL"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/29_text_Ed_Sheeran.png
try:
    _c29 = get_crop(29, 1032, 144)
    canvas.paste(_c29, (216, 120), _c29)
except Exception:
    pass
layout["Ed_Sheeran"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/30_text_Top_results.png
try:
    _c30 = get_crop(30, 295, 72)
    canvas.paste(_c30, (40, 373), _c30)
except Exception:
    pass
layout["Top_results"] = [40, 373, 335, 445]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/31_text_Performers.png
try:
    _c31 = get_crop(31, 293, 54)
    canvas.paste(_c31, (44, 1122), _c31)
except Exception:
    pass
layout["Performers"] = [44, 1122, 337, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/32_text_5_events.png
try:
    _c32 = get_crop(32, 173, 41)
    canvas.paste(_c32, (237, 1319), _c32)
except Exception:
    pass
layout["5_events"] = [237, 1319, 410, 1360]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/33_text_Events.png
try:
    _c33 = get_crop(33, 181, 57)
    canvas.paste(_c33, (43, 1868), _c33)
except Exception:
    pass
layout["Events"] = [43, 1868, 224, 1925]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/34_text_Venues.png
try:
    _c34 = get_crop(34, 197, 60)
    canvas.paste(_c34, (42, 2612), _c34)
except Exception:
    pass
layout["Venues"] = [42, 2612, 239, 2672]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/35_text_Sheraton_Grand_Hotel.png
try:
    _c35 = get_crop(35, 288, 162)
    canvas.paste(_c35, (288, 2792), _c35)
except Exception:
    pass
layout["Sheraton_Grand_Hotel"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/36_clickable_Browse.png
try:
    _c36 = get_crop(36, 288, 168)
    canvas.paste(_c36, (0, 2792), _c36)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/37_clickable_Tracking.png
try:
    _c37 = get_crop(37, 288, 168)
    canvas.paste(_c37, (864, 2792), _c37)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_04_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-7/38_clickable_Account.png
try:
    _c38 = get_crop(38, 288, 168)
    canvas.paste(_c38, (1152, 2792), _c38)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]
