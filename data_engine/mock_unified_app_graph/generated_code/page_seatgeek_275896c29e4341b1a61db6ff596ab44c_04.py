# page_id: page_seatgeek_275896c29e4341b1a61db6ff596ab44c_04
# screenshot: 2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7.png
# step_index: 4/9
# task: Open SeatGeek. Look up "Seattle Mariners" tickets. Select the next upcoming event in Los Angeles. Set quantity to 2 and select the best value tickets. Which section are tickets in?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background with a very light gray (matches app background)
draw.rectangle((0, 0, 1440, 2960), fill=(250, 250, 250))

# Status bar (top area) - slightly darker to emulate device bar
status_h = 80
draw.rectangle((0, 0, 1440, status_h), fill=(238, 238, 238))
# subtle bottom divider under status bar
draw.line((20, status_h - 1, 1420, status_h - 1), fill=(225, 225, 225), width=1)

# Search header / toolbar background (rounded search bar container)
search_top = 92
search_bottom = 172
search_margin = 32
draw.rounded_rectangle(
    (search_margin, search_top, 1440 - search_margin, search_bottom),
    radius=20,
    fill=(255, 255, 255),
    outline=(224, 224, 224),
    width=1
)

# Thin divider under the header / search area
draw.line((24, search_bottom + 8, 1416, search_bottom + 8), fill=(235, 235, 235), width=1)

# Top Results card background (group container)
top_results_top = 216
top_results_bottom = 494
draw.rounded_rectangle(
    (24, top_results_top, 1440 - 24, top_results_bottom),
    radius=12,
    fill=(255, 255, 255),
    outline=None
)
# internal separators for rows inside Top Results (approx positions)
draw.line((48, top_results_top + 96, 1392, top_results_top + 96), fill=(240, 240, 240), width=1)

# Section divider below Top Results
draw.line((24, top_results_bottom + 10, 1416, top_results_bottom + 10), fill=(235, 235, 235), width=1)

# Performers card background
performers_top = 1068
performers_bottom = 1228
draw.rounded_rectangle(
    (24, performers_top, 1440 - 24, performers_bottom),
    radius=12,
    fill=(255, 255, 255),
    outline=None
)
# subtle internal bottom line (separating header area from list)
draw.line((48, performers_bottom - 1, 1392, performers_bottom - 1), fill=(240, 240, 240), width=1)

# Events card background (list of events)
events_top = 1440
events_bottom = 1840
draw.rounded_rectangle(
    (24, events_top, 1440 - 24, events_bottom),
    radius=12,
    fill=(255, 255, 255),
    outline=None
)
# separators for three event rows (approx heights ~170 each)
draw.line((48, events_top + 170, 1392, events_top + 170), fill=(240, 240, 240), width=1)
draw.line((48, events_top + 340, 1392, events_top + 340), fill=(240, 240, 240), width=1)

# Another divider below events
draw.line((24, events_bottom + 10, 1416, events_bottom + 10), fill=(235, 235, 235), width=1)

# Recent searches card background
recent_top = 2200
recent_bottom = 2720
draw.rounded_rectangle(
    (24, recent_top, 1440 - 24, recent_bottom),
    radius=12,
    fill=(255, 255, 255),
    outline=None
)
# internal separators for recent search rows (approx positions)
# assume three recent items stacked
row_h = 180
draw.line((48, recent_top + row_h, 1392, recent_top + row_h), fill=(240, 240, 240), width=1)
draw.line((48, recent_top + 2 * row_h, 1392, recent_top + 2 * row_h), fill=(240, 240, 240), width=1)

# Light accent banner behind a top large header/title area (subtle, behind "Seattle Mariners" heading)
# place as a very faint off-white strip to differentiate header area
banner_top = 120
banner_bottom = 220
draw.rectangle((0, banner_top, 1440, banner_bottom), fill=(252, 252, 252))

# Separator lines across the full width at logical breakpoints (matching the visual rhythm)
separators = [top_results_bottom + 110, performers_top - 40, events_top - 40, recent_top - 40]
for y in separators:
    if 0 < y < 2792:  # avoid drawing over bottom nav area
        draw.line((20, y, 1420, y), fill=(238, 238, 238), width=1)

# Bottom navigation bar background area (leave icons/text to be pasted on top)
nav_top = 2792
draw.rectangle((0, nav_top, 1440, 2960), fill=(255, 255, 255))
# top border for nav bar
draw.line((24, nav_top, 1416, nav_top), fill=(230, 230, 230), width=1)
# slight shadow above nav to give depth
draw.line((24, nav_top - 2, 1416, nav_top - 2), fill=(245, 245, 245), width=1)

# Very subtle drop shadows for list cards (soft rectangular shadows below each major card)
shadow_color = (240, 240, 240)
# shadow under Top Results
draw.rectangle((24, top_results_bottom + 2, 1440 - 24, top_results_bottom + 6), fill=shadow_color)
# shadow under Performers
draw.rectangle((24, performers_bottom + 2, 1440 - 24, performers_bottom + 6), fill=shadow_color)
# shadow under Events
draw.rectangle((24, events_bottom + 2, 1440 - 24, events_bottom + 6), fill=shadow_color)
# shadow under Recent searches
draw.rectangle((24, recent_bottom + 2, 1440 - 24, recent_bottom + 6), fill=shadow_color)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/00_icon_Bruno_Mars.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 2519), _c0)
except Exception:
    pass
layout["Bruno_Mars"] = [0, 2519, 1440, 2687]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/01_icon_Seattle_WA.png
try:
    _c1 = get_crop(1, 1440, 179)
    canvas.paste(_c1, (0, 1963), _c1)
except Exception:
    pass
layout["Seattle,_WA"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 58, 61)
    canvas.paste(_c2, (245, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [245, 3, 303, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 43, 70)
    canvas.paste(_c3, (1155, 0), _c3)
except Exception:
    pass
layout["icon_3"] = [1155, 0, 1198, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/04_icon_Arlington_TX.png
try:
    _c4 = get_crop(4, 1440, 179)
    canvas.paste(_c4, (0, 1605), _c4)
except Exception:
    pass
layout["Arlington,_TX"] = [0, 1605, 1440, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/05_icon_Seattle_Mariners.png
try:
    _c5 = get_crop(5, 1440, 179)
    canvas.paste(_c5, (0, 1217), _c5)
except Exception:
    pass
layout["Seattle_Mariners"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/06_icon_Seattle_Mariners_at_Texas_Rangers.png
try:
    _c6 = get_crop(6, 1440, 179)
    canvas.paste(_c6, (0, 471), _c6)
except Exception:
    pass
layout["Seattle_Mariners_at_Texas"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/07_icon_Seattle_WA.png
try:
    _c7 = get_crop(7, 1440, 179)
    canvas.paste(_c7, (0, 829), _c7)
except Exception:
    pass
layout["Seattle,_WA"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/08_icon_7.48_Wy.png
try:
    _c8 = get_crop(8, 168, 144)
    canvas.paste(_c8, (48, 120), _c8)
except Exception:
    pass
layout["7.48_Wy"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 54, 61)
    canvas.paste(_c9, (315, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [315, 3, 369, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 93, 69)
    canvas.paste(_c10, (1219, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1219, 0, 1312, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/11_icon_Account.png
try:
    _c11 = get_crop(11, 288, 168)
    canvas.paste(_c11, (1152, 2792), _c11)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/12_icon_7.48_Wy.png
try:
    _c12 = get_crop(12, 44, 61)
    canvas.paste(_c12, (187, 2), _c12)
except Exception:
    pass
layout["7.48_Wy"] = [187, 2, 231, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/13_icon_Madison_Square_Garden.png
try:
    _c13 = get_crop(13, 1440, 168)
    canvas.paste(_c13, (0, 2351), _c13)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 2351, 1440, 2519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 45, 66)
    canvas.paste(_c14, (1326, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [1326, 2, 1371, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/15_icon_Clear.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1248, 120), _c15)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/16_icon_Arlington_TX.png
try:
    _c16 = get_crop(16, 1440, 179)
    canvas.paste(_c16, (0, 650), _c16)
except Exception:
    pass
layout["Arlington,_TX"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/17_icon_Tue.png
try:
    _c17 = get_crop(17, 1440, 179)
    canvas.paste(_c17, (0, 1963), _c17)
except Exception:
    pass
layout["Tue,"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/18_icon_Tomorrow.png
try:
    _c18 = get_crop(18, 1440, 179)
    canvas.paste(_c18, (0, 650), _c18)
except Exception:
    pass
layout["Tomorrow"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/19_icon_Wed.png
try:
    _c19 = get_crop(19, 1440, 179)
    canvas.paste(_c19, (0, 1784), _c19)
except Exception:
    pass
layout["Wed,"] = [0, 1784, 1440, 1963]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/20_icon_7.48_Wy.png
try:
    _c20 = get_crop(20, 52, 62)
    canvas.paste(_c20, (117, 1), _c20)
except Exception:
    pass
layout["7.48_Wy"] = [117, 1, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/21_icon_Wed.png
try:
    _c21 = get_crop(21, 1440, 179)
    canvas.paste(_c21, (0, 829), _c21)
except Exception:
    pass
layout["Wed,"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/22_text_Seattle_Mariners.png
try:
    _c22 = get_crop(22, 1032, 144)
    canvas.paste(_c22, (216, 120), _c22)
except Exception:
    pass
layout["Seattle_Mariners"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/23_text_Top_results.png
try:
    _c23 = get_crop(23, 295, 72)
    canvas.paste(_c23, (40, 373), _c23)
except Exception:
    pass
layout["Top_results"] = [40, 373, 335, 445]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/24_text_Performers.png
try:
    _c24 = get_crop(24, 293, 54)
    canvas.paste(_c24, (44, 1122), _c24)
except Exception:
    pass
layout["Performers"] = [44, 1122, 337, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/25_text_Events.png
try:
    _c25 = get_crop(25, 177, 54)
    canvas.paste(_c25, (46, 1510), _c25)
except Exception:
    pass
layout["Events"] = [46, 1510, 223, 1564]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/26_text_Recent_searches.png
try:
    _c26 = get_crop(26, 436, 57)
    canvas.paste(_c26, (44, 2257), _c26)
except Exception:
    pass
layout["Recent_searches"] = [44, 2257, 480, 2314]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/27_text_Bruno_Mars.png
try:
    _c27 = get_crop(27, 254, 54)
    canvas.paste(_c27, (237, 2579), _c27)
except Exception:
    pass
layout["Bruno_Mars"] = [237, 2579, 491, 2633]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/28_text_L_Olympia_Olympia_Theatre.png
try:
    _c28 = get_crop(28, 1440, 105)
    canvas.paste(_c28, (0, 2687), _c28)
except Exception:
    pass
layout["L'Olympia_(Olympia_Theatr"] = [0, 2687, 1440, 2792]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/29_clickable_Browse.png
try:
    _c29 = get_crop(29, 288, 168)
    canvas.paste(_c29, (0, 2792), _c29)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/30_clickable_Search.png
try:
    _c30 = get_crop(30, 288, 162)
    canvas.paste(_c30, (288, 2792), _c30)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/31_clickable_Tickets.png
try:
    _c31 = get_crop(31, 288, 168)
    canvas.paste(_c31, (576, 2792), _c31)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_04_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-7/32_clickable_Tracking.png
try:
    _c32 = get_crop(32, 288, 168)
    canvas.paste(_c32, (864, 2792), _c32)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]
