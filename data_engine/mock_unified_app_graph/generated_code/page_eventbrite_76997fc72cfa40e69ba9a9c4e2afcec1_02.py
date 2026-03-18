# page_id: page_eventbrite_76997fc72cfa40e69ba9a9c4e2afcec1_02
# screenshot: 2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4.png
# step_index: 2/3
# task: Open Eventbrite. Open favorite tab and remove the second event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill
draw.rectangle((0, 0, 1440, 2960), fill="#FFFFFF")

# Status bar (top) background
status_h = 56
draw.rectangle((0, 0, 1440, status_h), fill="#CFCFCF")

# Thin divider under status bar
draw.line((0, status_h, 1440, status_h), fill="#BFBFBF", width=1)

# Header underline / subtle toolbar divider (below title area)
toolbar_div_y = 260
draw.line((24, toolbar_div_y, 1440-24, toolbar_div_y), fill="#EFEFF1", width=1)

# Tabs underline: "Events" selected indicator (blue strip)
tabs_underline_x = 24
tabs_underline_w = 160
tabs_underline_y = 420
draw.rectangle((tabs_underline_x, tabs_underline_y, tabs_underline_x + tabs_underline_w, tabs_underline_y + 4), fill="#2F6DF6")

# Section card rounded backgrounds (light card outlines for each event row)
card_margin_x = 24
card_radius = 12
cards = [
    (675, 1071),
    (1071, 1467),
    (1668, 2064),
    (2265, 2661)
]
for top, bottom in cards:
    # Slightly off-white card fill to separate from page white
    draw.rounded_rectangle(
        (card_margin_x, top + 6, 1440 - card_margin_x, bottom - 6),
        radius=card_radius,
        fill="#FFFFFF",
        outline="#EEEEF1",
        width=1
    )
    # subtle inner shadow line at top of card
    draw.line((card_margin_x + 1, top + 7, 1440 - card_margin_x - 1, top + 7), fill="#F6F6F8", width=1)

    # left thumbnail/background area for image posts (background only, actual image will be pasted)
    thumb_x = card_margin_x + 12
    thumb_y = top + 24
    thumb_w = 128
    thumb_h = 128
    draw.rounded_rectangle(
        (thumb_x, thumb_y, thumb_x + thumb_w, thumb_y + thumb_h),
        radius=8,
        fill="#F2F3F5",
        outline="#E6E7EA",
        width=1
    )

# Separator lines between major sections / date groups
separators = [1071, 1668, 2265, 2661]
for y in separators:
    draw.line((24, y, 1440 - 24, y), fill="#F0F0F2", width=1)

# Large section headings area spacing divider (below page title area)
heading_div_y = 520
draw.line((24, heading_div_y, 1440 - 24, heading_div_y), fill="#FFFFFF", width=1)  # nearly invisible spacer

# Bottom navigation bar background and top divider
nav_top = 2804
draw.rectangle((0, nav_top, 1440, 2960), fill="#FFFFFF")
draw.line((0, nav_top, 1440, nav_top), fill="#E8E8EA", width=2)

# Slight top shadow for nav bar to lift it
shadow_y1 = nav_top
shadow_y2 = nav_top + 6
for i in range(6):
    alpha = int(18 - i*3)
    if alpha < 0:
        alpha = 0
    # simulate shadow by drawing diminishing grey lines
    shade = 230 + i
    draw.line((0, shadow_y1 + i, 1440, shadow_y1 + i), fill=(shade, shade, shade), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/00_icon_SHIPYARD.png
try:
    _c0 = get_crop(0, 1440, 396)
    canvas.paste(_c0, (0, 1668), _c0)
except Exception:
    pass
layout["SHIPYARD"] = [0, 1668, 1440, 2064]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/01_icon_6827_Niorf.png
try:
    _c1 = get_crop(1, 1440, 396)
    canvas.paste(_c1, (0, 1071), _c1)
except Exception:
    pass
layout["6827_Niorf"] = [0, 1071, 1440, 1467]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/02_icon_Win_More_Business.png
try:
    _c2 = get_crop(2, 1440, 396)
    canvas.paste(_c2, (0, 675), _c2)
except Exception:
    pass
layout["&_Win_More_Business"] = [0, 675, 1440, 1071]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/03_icon_JACK.png
try:
    _c3 = get_crop(3, 1440, 396)
    canvas.paste(_c3, (0, 2265), _c3)
except Exception:
    pass
layout["JACK"] = [0, 2265, 1440, 2661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/04_icon_Maouc.png
try:
    _c4 = get_crop(4, 288, 156)
    canvas.paste(_c4, (288, 2804), _c4)
except Exception:
    pass
layout["Maouc"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/05_icon_Events.png
try:
    _c5 = get_crop(5, 220, 133)
    canvas.paste(_c5, (0, 344), _c5)
except Exception:
    pass
layout["Events"] = [0, 344, 220, 477]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/06_icon_Favorites.png
try:
    _c6 = get_crop(6, 308, 144)
    canvas.paste(_c6, (217, 330), _c6)
except Exception:
    pass
layout["Favorites"] = [217, 330, 525, 474]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/07_icon_Organizers.png
try:
    _c7 = get_crop(7, 308, 144)
    canvas.paste(_c7, (217, 330), _c7)
except Exception:
    pass
layout["Organizers"] = [217, 330, 525, 474]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 62, 62)
    canvas.paste(_c8, (310, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [310, 1, 372, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/09_icon_5.37.png
try:
    _c9 = get_crop(9, 61, 64)
    canvas.paste(_c9, (179, 1), _c9)
except Exception:
    pass
layout["5.37"] = [179, 1, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/10_icon_5.37.png
try:
    _c10 = get_crop(10, 64, 66)
    canvas.paste(_c10, (111, 0), _c10)
except Exception:
    pass
layout["5.37"] = [111, 0, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/11_icon_Tickets.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (864, 2804), _c11)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 52, 62)
    canvas.paste(_c12, (249, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [249, 1, 301, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/13_icon_Unlike_event.png
try:
    _c13 = get_crop(13, 72, 72)
    canvas.paste(_c13, (1320, 1347), _c13)
except Exception:
    pass
layout["Unlike_event"] = [1320, 1347, 1392, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 56, 71)
    canvas.paste(_c14, (1316, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1316, 0, 1372, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/15_icon_Unlike_event.png
try:
    _c15 = get_crop(15, 72, 72)
    canvas.paste(_c15, (1320, 1944), _c15)
except Exception:
    pass
layout["Unlike_event"] = [1320, 1944, 1392, 2016]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/16_icon_Unlike_event.png
try:
    _c16 = get_crop(16, 72, 72)
    canvas.paste(_c16, (1320, 2541), _c16)
except Exception:
    pass
layout["Unlike_event"] = [1320, 2541, 1392, 2613]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/17_icon_Share_Event.png
try:
    _c17 = get_crop(17, 72, 72)
    canvas.paste(_c17, (1200, 1944), _c17)
except Exception:
    pass
layout["Share_Event"] = [1200, 1944, 1272, 2016]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 90, 71)
    canvas.paste(_c18, (1210, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1210, 0, 1300, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/19_icon_Unlike_event.png
try:
    _c19 = get_crop(19, 72, 72)
    canvas.paste(_c19, (1320, 951), _c19)
except Exception:
    pass
layout["Unlike_event"] = [1320, 951, 1392, 1023]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/20_icon_Share_Event.png
try:
    _c20 = get_crop(20, 72, 72)
    canvas.paste(_c20, (1200, 951), _c20)
except Exception:
    pass
layout["Share_Event"] = [1200, 951, 1272, 1023]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/21_icon_Share_Event.png
try:
    _c21 = get_crop(21, 72, 72)
    canvas.paste(_c21, (1200, 2541), _c21)
except Exception:
    pass
layout["Share_Event"] = [1200, 2541, 1272, 2613]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/22_icon_Share_Event.png
try:
    _c22 = get_crop(22, 72, 72)
    canvas.paste(_c22, (1200, 1347), _c22)
except Exception:
    pass
layout["Share_Event"] = [1200, 1347, 1272, 1419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 42, 67)
    canvas.paste(_c23, (1272, 1), _c23)
except Exception:
    pass
layout["icon_23"] = [1272, 1, 1314, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/24_icon_Shipyard_Open_Studios_-_Spring_2024.png
try:
    _c24 = get_crop(24, 1440, 396)
    canvas.paste(_c24, (0, 1668), _c24)
except Exception:
    pass
layout["Shipyard_Open_Studios_-_S"] = [0, 1668, 1440, 2064]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/25_icon_AAcn.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (0, 2804), _c25)
except Exception:
    pass
layout["AAcn"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/26_icon_THE_LIFE_DEATH_OF_ART.png
try:
    _c26 = get_crop(26, 1440, 396)
    canvas.paste(_c26, (0, 2265), _c26)
except Exception:
    pass
layout["THE_LIFE_&_DEATH_OF_ART"] = [0, 2265, 1440, 2661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/27_icon_THE_LIFE_DEATH_OF_ART.png
try:
    _c27 = get_crop(27, 1440, 396)
    canvas.paste(_c27, (0, 2265), _c27)
except Exception:
    pass
layout["THE_LIFE_&_DEATH_OF_ART"] = [0, 2265, 1440, 2661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/28_icon_Realtors_Refine_Client_Appreciation.png
try:
    _c28 = get_crop(28, 1440, 396)
    canvas.paste(_c28, (0, 675), _c28)
except Exception:
    pass
layout["Realtors:_Refine_Client_A"] = [0, 675, 1440, 1071]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/29_icon_More.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (1152, 2804), _c29)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/30_icon_Art_Rangers_Jazz_Night_Auction.png
try:
    _c30 = get_crop(30, 1440, 396)
    canvas.paste(_c30, (0, 1071), _c30)
except Exception:
    pass
layout["Art_Rangers_Jazz_Night_&_"] = [0, 1071, 1440, 1467]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/31_icon_Shipyard_Open_Studios_-_Spring_2024.png
try:
    _c31 = get_crop(31, 1440, 396)
    canvas.paste(_c31, (0, 1668), _c31)
except Exception:
    pass
layout["Shipyard_Open_Studios_-_S"] = [0, 1668, 1440, 2064]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/32_icon_5.37.png
try:
    _c32 = get_crop(32, 93, 61)
    canvas.paste(_c32, (15, 2), _c32)
except Exception:
    pass
layout["5.37"] = [15, 2, 108, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/33_icon_Maouc.png
try:
    _c33 = get_crop(33, 288, 156)
    canvas.paste(_c33, (576, 2804), _c33)
except Exception:
    pass
layout["Maouc"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/34_icon_icon_34.png
try:
    _c34 = get_crop(34, 56, 63)
    canvas.paste(_c34, (381, 1), _c34)
except Exception:
    pass
layout["icon_34"] = [381, 1, 437, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/35_text_Today.png
try:
    _c35 = get_crop(35, 185, 85)
    canvas.paste(_c35, (37, 576), _c35)
except Exception:
    pass
layout["Today"] = [37, 576, 222, 661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/36_text_Wed.png
try:
    _c36 = get_crop(36, 151, 73)
    canvas.paste(_c36, (263, 582), _c36)
except Exception:
    pass
layout["Wed,"] = [263, 582, 414, 655]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/37_text_24.png
try:
    _c37 = get_crop(37, 84, 61)
    canvas.paste(_c37, (519, 584), _c37)
except Exception:
    pass
layout["24"] = [519, 584, 603, 645]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/38_text_Wed.png
try:
    _c38 = get_crop(38, 110, 54)
    canvas.paste(_c38, (390, 727), _c38)
except Exception:
    pass
layout["Wed,"] = [390, 727, 500, 781]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/39_text_24_._9_00_PM_EDT.png
try:
    _c39 = get_crop(39, 1440, 396)
    canvas.paste(_c39, (0, 675), _c39)
except Exception:
    pass
layout["24_._9:00_PM_EDT"] = [0, 675, 1440, 1071]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/40_text_Strategies_to_Wow_Win_More_Business.png
try:
    _c40 = get_crop(40, 1440, 396)
    canvas.paste(_c40, (0, 675), _c40)
except Exception:
    pass
layout["Strategies_to_Wow_&_Win_M"] = [0, 675, 1440, 1071]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/41_text_JACK.png
try:
    _c41 = get_crop(41, 105, 45)
    canvas.paste(_c41, (390, 945), _c41)
except Exception:
    pass
layout["JACK"] = [390, 945, 495, 990]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/42_text_Wed_Apr_24_-.png
try:
    _c42 = get_crop(42, 249, 52)
    canvas.paste(_c42, (393, 1126), _c42)
except Exception:
    pass
layout["Wed,_Apr_24_-"] = [393, 1126, 642, 1178]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/43_text_7_00_PM_PDT.png
try:
    _c43 = get_crop(43, 1440, 396)
    canvas.paste(_c43, (0, 1071), _c43)
except Exception:
    pass
layout["7:00_PM_PDT"] = [0, 1071, 1440, 1467]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/44_text_The_Faight_Collective.png
try:
    _c44 = get_crop(44, 1440, 396)
    canvas.paste(_c44, (0, 1071), _c44)
except Exception:
    pass
layout["The_Faight_Collective"] = [0, 1071, 1440, 1467]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/45_text_Apr_27.png
try:
    _c45 = get_crop(45, 191, 80)
    canvas.paste(_c45, (159, 1573), _c45)
except Exception:
    pass
layout["Apr_27"] = [159, 1573, 350, 1653]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/46_text_Hunters_Point_Shipyard.png
try:
    _c46 = get_crop(46, 1440, 396)
    canvas.paste(_c46, (0, 1668), _c46)
except Exception:
    pass
layout["Hunters_Point_Shipyard"] = [0, 1668, 1440, 2064]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/47_text_Sun.png
try:
    _c47 = get_crop(47, 141, 76)
    canvas.paste(_c47, (39, 2170), _c47)
except Exception:
    pass
layout["Sun,"] = [39, 2170, 180, 2246]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/48_text_28.png
try:
    _c48 = get_crop(48, 84, 63)
    canvas.paste(_c48, (284, 2173), _c48)
except Exception:
    pass
layout["28"] = [284, 2173, 368, 2236]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_02_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-4/49_text_JACK.png
try:
    _c49 = get_crop(49, 107, 48)
    canvas.paste(_c49, (389, 2465), _c49)
except Exception:
    pass
layout["JACK"] = [389, 2465, 496, 2513]
