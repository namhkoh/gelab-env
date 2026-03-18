# page_id: page_eventbrite_def19c5d5bd0474abe83d89af89419b3_06
# screenshot: 2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8.png
# step_index: 6/8
# task: Open Eventbrite. Set the city to Los Angeles. Select the second recommendation on the home tab. Follow the organizer and look for the time and date of the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background/base fill
draw.rectangle([0, 0, 1440, 2960], fill="#FAFAFB")  # subtle off-white overall background

# Status bar area (top)
status_h = 72
draw.rectangle([0, 0, 1440, status_h], fill="#CFCFCF")  # light gray status bar
draw.line([(0, status_h), (1440, status_h)], fill="#B7B7B7", width=1)  # divider under status bar

# Header / toolbar area behind the search field (kept minimalist so icons/text will be pasted on top)
header_top = status_h
header_bottom = 240
draw.rectangle([0, header_top, 1440, header_bottom], fill="#FFFFFF")
# subtle bottom divider shadow for the header
draw.line([(24, header_bottom), (1440 - 24, header_bottom)], fill="#EFEFF1", width=2)

# Function to draw a card with subtle shadow
def draw_card(x, y, w, h, radius=20, fill="#FFFFFF", shadow_color="#E9E9EA", outline="#F0F0F2"):
    # shadow (offset down-right)
    shadow_offset = 6
    draw.rounded_rectangle(
        [x + shadow_offset, y + shadow_offset, x + w + shadow_offset, y + h + shadow_offset],
        radius=radius,
        fill=shadow_color,
        outline=None
    )
    # card background
    draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=fill, outline=outline, width=1)

# Event/listing cards (backgrounds behind each detected row)
cards = [
    (48, 490, 1344, 396),
    (48, 886, 1344, 396),
    (48, 1282, 1344, 396),
    (48, 1678, 1344, 396),
    (48, 2074, 1344, 396),
    (48, 2470, 1344, 346),
]
for (x, y, w, h) in cards:
    draw_card(x, y, w, h, radius=16, fill="#FFFFFF", shadow_color="#F1F1F2", outline="#F3F3F5")

# Thin separators between rows (subtle)
for (x, y, w, h) in cards:
    sep_y = y + h + 12
    draw.line([(x + 12, sep_y), (x + w - 12, sep_y)], fill="#F4F4F6", width=1)

# Floating location pill background (placed behind detected "Los Angeles" widget)
pill_x, pill_y, pill_w, pill_h = 492, 2651, 456, 117
# small shadow
draw.rounded_rectangle(
    [pill_x + 4, pill_y + 6, pill_x + pill_w + 4, pill_y + pill_h + 6],
    radius=36, fill="#EDEFF2"
)
# pill base
draw.rounded_rectangle([pill_x, pill_y, pill_x + pill_w, pill_y + pill_h], radius=36, fill="#FFFFFF", outline="#E8E8EA", width=1)

# Bottom navigation bar area
nav_top = 2804
nav_bottom = 2960
draw.rectangle([0, nav_top, 1440, nav_bottom], fill="#FFFFFF")
draw.line([(0, nav_top), (1440, nav_top)], fill="#EDEEF0", width=2)

# Small top shadow under the last content area to lift it above navigation
draw.rectangle([24, nav_top - 8, 1440 - 24, nav_top - 6], fill="#F6F6F7")

# Final gentle vertical rhythm lines on the left to indicate content flow (very subtle)
for yi in range(430, 2600, 400):
    draw.line([(48, yi), (48, yi + 320)], fill="#FFFFFF", width=6)  # ensures left edge stays visually clean

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/00_icon_Q_Search_events.png
try:
    _c0 = get_crop(0, 1179, 144)
    canvas.paste(_c0, (195, 93), _c0)
except Exception:
    pass
layout["Q_Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/01_icon_FRIDAY.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 2074), _c1)
except Exception:
    pass
layout["FRIDAY"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/02_icon_NDIE_DANCEPA.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1678), _c2)
except Exception:
    pass
layout["NDIE_DANCEPA"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/03_icon_Ibaigktsinel.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1282), _c3)
except Exception:
    pass
layout["Ibaigktsinel"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/04_icon_Los_Angeles.png
try:
    _c4 = get_crop(4, 456, 117)
    canvas.paste(_c4, (492, 2651), _c4)
except Exception:
    pass
layout["Los_Angeles"] = [492, 2651, 948, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 139)
    canvas.paste(_c5, (1140, 1935), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 123)
    canvas.paste(_c6, (1140, 1555), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1555, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/07_icon_NDIE.png
try:
    _c7 = get_crop(7, 1344, 396)
    canvas.paste(_c7, (48, 886), _c7)
except Exception:
    pass
layout["NDIE"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 49, 65)
    canvas.paste(_c8, (1153, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [1153, 2, 1202, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/09_icon_Afliccion_Perdida_y.png
try:
    _c9 = get_crop(9, 144, 123)
    canvas.paste(_c9, (1140, 2347), _c9)
except Exception:
    pass
layout["Afliccion,_Perdida_y"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/10_icon_Overflow_menu_button.png
try:
    _c10 = get_crop(10, 144, 139)
    canvas.paste(_c10, (1284, 1935), _c10)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 123)
    canvas.paste(_c11, (1284, 1555), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1555, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 123)
    canvas.paste(_c12, (1284, 2347), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/13_icon_Sylmai.png
try:
    _c13 = get_crop(13, 288, 156)
    canvas.paste(_c13, (288, 2804), _c13)
except Exception:
    pass
layout["Sylmai"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/14_icon_Club_Decades.png
try:
    _c14 = get_crop(14, 144, 139)
    canvas.paste(_c14, (1140, 1143), _c14)
except Exception:
    pass
layout["Club_Decades"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 139)
    canvas.paste(_c15, (1284, 1143), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/16_icon_4721_creator_followers.png
try:
    _c16 = get_crop(16, 1344, 396)
    canvas.paste(_c16, (48, 886), _c16)
except Exception:
    pass
layout["4721_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/17_icon_Favorite_button.png
try:
    _c17 = get_crop(17, 144, 123)
    canvas.paste(_c17, (1140, 763), _c17)
except Exception:
    pass
layout["Favorite_button"] = [1140, 763, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/18_icon_8_60_creator_followers.png
try:
    _c18 = get_crop(18, 1344, 396)
    canvas.paste(_c18, (48, 1678), _c18)
except Exception:
    pass
layout["8_60_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/19_icon_Home.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (0, 2804), _c19)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 97, 60)
    canvas.paste(_c20, (1216, 3), _c20)
except Exception:
    pass
layout["icon_20"] = [1216, 3, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 60, 59)
    canvas.paste(_c21, (312, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [312, 3, 372, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/22_icon_Overflow_menu_button.png
try:
    _c22 = get_crop(22, 144, 123)
    canvas.paste(_c22, (1284, 763), _c22)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 763, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/23_icon_5.35.png
try:
    _c23 = get_crop(23, 57, 60)
    canvas.paste(_c23, (182, 2), _c23)
except Exception:
    pass
layout["5.35"] = [182, 2, 239, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 52, 60)
    canvas.paste(_c24, (247, 2), _c24)
except Exception:
    pass
layout["icon_24"] = [247, 2, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/25_icon_5.35.png
try:
    _c25 = get_crop(25, 102, 99)
    canvas.paste(_c25, (41, 122), _c25)
except Exception:
    pass
layout["5.35"] = [41, 122, 143, 221]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/26_icon_ter_for_Break_Into_Tech_nowl.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 490), _c26)
except Exception:
    pass
layout["ter_for_Break_Into_Tech_n"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/27_icon_8_21126_creator_followers.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 1282), _c27)
except Exception:
    pass
layout["8_21126_creator_followers"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 48, 53)
    canvas.paste(_c28, (1321, 7), _c28)
except Exception:
    pass
layout["icon_28"] = [1321, 7, 1369, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/29_icon_Public_House_Los_Angeles_CA.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 490), _c29)
except Exception:
    pass
layout["Public_House_(Los_Angeles"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/30_icon_5.35.png
try:
    _c30 = get_crop(30, 58, 61)
    canvas.paste(_c30, (115, 2), _c30)
except Exception:
    pass
layout["5.35"] = [115, 2, 173, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/31_icon_Q_Search_events.png
try:
    _c31 = get_crop(31, 44, 57)
    canvas.paste(_c31, (385, 6), _c31)
except Exception:
    pass
layout["Q_Search_events"] = [385, 6, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/32_icon_Free.png
try:
    _c32 = get_crop(32, 1344, 346)
    canvas.paste(_c32, (48, 2470), _c32)
except Exception:
    pass
layout["Free"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/33_icon_Free.png
try:
    _c33 = get_crop(33, 128, 74)
    canvas.paste(_c33, (245, 1748), _c33)
except Exception:
    pass
layout["Free"] = [245, 1748, 373, 1822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/34_icon_9.30_PM_PDT.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 886), _c34)
except Exception:
    pass
layout["9.30_PM_PDT"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/35_icon_Punk_Indie_Rock_Dance_Party.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 2074), _c35)
except Exception:
    pass
layout["Punk;_Indie_Rock_Dance_Pa"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/36_icon_Tickets.png
try:
    _c36 = get_crop(36, 288, 156)
    canvas.paste(_c36, (864, 2804), _c36)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/37_icon_YEAH_YEAH_YAS_Queer_Indie_Dance_Party_LA.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 1678), _c37)
except Exception:
    pass
layout["YEAH_YEAH_YAS:_Queer_Indi"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/38_icon_5.30_PM_PDT.png
try:
    _c38 = get_crop(38, 1344, 346)
    canvas.paste(_c38, (48, 2470), _c38)
except Exception:
    pass
layout["5.30_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/39_icon_31_creator_followers.png
try:
    _c39 = get_crop(39, 288, 156)
    canvas.paste(_c39, (576, 2804), _c39)
except Exception:
    pass
layout["31_creator_followers"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/40_text_5.35.png
try:
    _c40 = get_crop(40, 92, 43)
    canvas.paste(_c40, (22, 17), _c40)
except Exception:
    pass
layout["5.35"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/41_text_More_events_you_II_love.png
try:
    _c41 = get_crop(41, 1344, 396)
    canvas.paste(_c41, (48, 490), _c41)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/42_text_Mon_May_13.png
try:
    _c42 = get_crop(42, 222, 43)
    canvas.paste(_c42, (393, 2525), _c42)
except Exception:
    pass
layout["Mon,_May_13"] = [393, 2525, 615, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/43_text_5.30_PM_PDT.png
try:
    _c43 = get_crop(43, 1344, 346)
    canvas.paste(_c43, (48, 2470), _c43)
except Exception:
    pass
layout["5.30_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/44_text_31_creator_followers.png
try:
    _c44 = get_crop(44, 1344, 346)
    canvas.paste(_c44, (48, 2470), _c44)
except Exception:
    pass
layout["31_creator_followers"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_06_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-8/45_clickable_More.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (1152, 2804), _c45)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
