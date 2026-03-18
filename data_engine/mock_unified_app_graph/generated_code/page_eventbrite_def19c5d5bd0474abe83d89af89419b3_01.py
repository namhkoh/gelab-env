# page_id: page_eventbrite_def19c5d5bd0474abe83d89af89419b3_01
# screenshot: 2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3.png
# step_index: 1/8
# task: Open Eventbrite. Set the city to Los Angeles. Select the second recommendation on the home tab. Follow the organizer and look for the time and date of the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint the general page background and structural UI chrome for the Eventbrite-like screen.
# Uses provided variables: canvas (Image), draw (ImageDraw), font_sm/font_md/font_lg/font_xl

# Full background (very light off-white to match screenshot)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFAFC")

# Status bar area at top (~56px) - muted grey
status_h = 56
draw.rectangle([(0, 0), (1440, status_h)], fill="#BDBDBD")

# Header / toolbar area under status bar
header_top = status_h
header_bottom = 200
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")
# subtle bottom divider under header
draw.line([(48, header_bottom), (1392, header_bottom)], fill="#E9E8EB", width=1)

# Main content column inset
content_x = 48
content_w = 1344  # as provided by detections

# Card definitions: (y_top, height)
card_defs = [
    (490, 396),
    (886, 396),
    (1282, 396),
    (1678, 396),
    (2074, 396),
    (2470, 346),  # last card shorter per detections
]

# Draw rounded card backgrounds and light separators
card_radius = 12
card_fill = "#FFFFFF"
card_outline = "#F1EEF3"  # faint outline for cards
separator_color = "#F3EFF4"

for y, h in card_defs:
    x0 = content_x
    y0 = y
    x1 = content_x + content_w
    y1 = y + h
    # background rounded rectangle
    draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=card_radius, fill=card_fill, outline=card_outline, width=1)
    # top separator line (to visually separate stacked cards)
    draw.line([(x0, y0), (x1, y0)], fill=separator_color, width=1)
    # subtle bottom divider slightly below card to mimic spacing
    draw.line([(x0, y1), (x1, y1)], fill="#F7F6F8", width=1)

# Content area subtle vertical guideline (not visible in final UI, helps align)
# Draw a very faint guide line along the left edge of content (keeps layout structure)
draw.line([(content_x, header_bottom+8), (content_x, 2760)], fill="#FCFBFD", width=1)

# Floating content/backdrop area near the mid-bottom where the "Find / Online events" pill would appear
# Draw only a subtle shadow strip (do not draw the pill itself which will be pasted)
floating_shadow_top = 2500
floating_shadow_bottom = 2560
draw.rectangle([(120, floating_shadow_top), (1320, floating_shadow_bottom)], fill="#FFFFFF", outline="#EFEAF0")
# tiny soft divider above floating area to ground it
draw.line([(120, floating_shadow_top), (1320, floating_shadow_top)], fill="#ECE9ED", width=1)

# Bottom navigation bar background and top border
nav_top = 2804
nav_bottom = 2960
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill="#FFFFFF")
draw.line([(0, nav_top), (1440, nav_top)], fill="#EDE9EB", width=1)

# A subtle top shadow for the whole content column to add depth (very light)
shadow_y = header_bottom + 6
draw.line([(48, shadow_y), (1392, shadow_y)], fill="#F6F4F7", width=1)

# Small left gutter shadow to separate edge from card thumbnail area
gutter_x = content_x + 16
draw.line([(gutter_x, header_bottom + 12), (gutter_x, nav_top - 12)], fill="#FBF9FB", width=1)

# End of structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/00_icon_Online.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 490), _c0)
except Exception:
    pass
layout["Online"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/01_icon_Online.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 2074), _c1)
except Exception:
    pass
layout["Online"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/02_icon_Online.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1678), _c2)
except Exception:
    pass
layout["Online"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/03_icon_Q_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Q_Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 125)
    canvas.paste(_c4, (1140, 2345), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1140, 2345, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 125)
    canvas.paste(_c5, (1140, 1949), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1949, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 125)
    canvas.paste(_c6, (1284, 2345), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2345, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 139)
    canvas.paste(_c7, (1284, 1539), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/08_icon_On..png
try:
    _c8 = get_crop(8, 288, 156)
    canvas.paste(_c8, (288, 2804), _c8)
except Exception:
    pass
layout["On."] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1284, 747), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/10_icon_Overflow_menu_button.png
try:
    _c10 = get_crop(10, 144, 125)
    canvas.paste(_c10, (1284, 1949), _c10)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1949, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/11_icon_5.34.png
try:
    _c11 = get_crop(11, 106, 101)
    canvas.paste(_c11, (39, 121), _c11)
except Exception:
    pass
layout["5.34"] = [39, 121, 145, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 139)
    canvas.paste(_c12, (1284, 1143), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/13_icon_5.34.png
try:
    _c13 = get_crop(13, 54, 61)
    canvas.paste(_c13, (184, 2), _c13)
except Exception:
    pass
layout["5.34"] = [184, 2, 238, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/14_icon_Home.png
try:
    _c14 = get_crop(14, 288, 156)
    canvas.paste(_c14, (0, 2804), _c14)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/15_icon_Online_events.png
try:
    _c15 = get_crop(15, 586, 117)
    canvas.paste(_c15, (427, 2651), _c15)
except Exception:
    pass
layout["Online_events"] = [427, 2651, 1013, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 60, 59)
    canvas.paste(_c16, (312, 3), _c16)
except Exception:
    pass
layout["icon_16"] = [312, 3, 372, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 51, 59)
    canvas.paste(_c17, (248, 3), _c17)
except Exception:
    pass
layout["icon_17"] = [248, 3, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/18_icon_Art_for_Grief_and_Loss.png
try:
    _c18 = get_crop(18, 1344, 396)
    canvas.paste(_c18, (48, 1282), _c18)
except Exception:
    pass
layout["Art_for_Grief_and_Loss"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/19_icon_Favorite_button.png
try:
    _c19 = get_crop(19, 144, 139)
    canvas.paste(_c19, (1140, 747), _c19)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/20_icon_Favorite_button.png
try:
    _c20 = get_crop(20, 144, 139)
    canvas.paste(_c20, (1140, 1143), _c20)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/21_icon_Favorite_button.png
try:
    _c21 = get_crop(21, 144, 139)
    canvas.paste(_c21, (1140, 1539), _c21)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 48, 53)
    canvas.paste(_c22, (1321, 7), _c22)
except Exception:
    pass
layout["icon_22"] = [1321, 7, 1369, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/23_icon_Tickets.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (864, 2804), _c23)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/24_icon_Working_with_Grief_and_Loss.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 490), _c24)
except Exception:
    pass
layout["Working_with_Grief_and_Lo"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/25_icon_5.34.png
try:
    _c25 = get_crop(25, 57, 60)
    canvas.paste(_c25, (116, 3), _c25)
except Exception:
    pass
layout["5.34"] = [116, 3, 173, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 67, 60)
    canvas.paste(_c26, (1212, 3), _c26)
except Exception:
    pass
layout["icon_26"] = [1212, 3, 1279, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/27_icon_S_00_AM_EDT.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 1678), _c27)
except Exception:
    pass
layout["S:00_AM_EDT"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/28_icon_5_O0_AM_EDT.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 2074), _c28)
except Exception:
    pass
layout["5:O0_AM_EDT"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/29_icon_Q_Search_events.png
try:
    _c29 = get_crop(29, 44, 56)
    canvas.paste(_c29, (385, 7), _c29)
except Exception:
    pass
layout["Q_Search_events"] = [385, 7, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/30_icon_suppoloyed_Orilee_herapeeticrarard_Outh_.png
try:
    _c30 = get_crop(30, 1344, 396)
    canvas.paste(_c30, (48, 1282), _c30)
except Exception:
    pass
layout["suppoloyed_Orilee__herape"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/31_icon_icon_31.png
try:
    _c31 = get_crop(31, 42, 56)
    canvas.paste(_c31, (1272, 5), _c31)
except Exception:
    pass
layout["icon_31"] = [1272, 5, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/32_icon_Understanding_Grief_and_Loss.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 886), _c32)
except Exception:
    pass
layout["Understanding_Grief_and_L"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/33_icon_Online.png
try:
    _c33 = get_crop(33, 112, 53)
    canvas.paste(_c33, (390, 1496), _c33)
except Exception:
    pass
layout["Online"] = [390, 1496, 502, 1549]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/34_icon_Online.png
try:
    _c34 = get_crop(34, 112, 54)
    canvas.paste(_c34, (390, 703), _c34)
except Exception:
    pass
layout["Online"] = [390, 703, 502, 757]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/35_icon_Art_for_Grief_and_Loss.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 1282), _c35)
except Exception:
    pass
layout["Art_for_Grief_and_Loss"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/36_icon_9_2273_creator_followers.png
try:
    _c36 = get_crop(36, 288, 156)
    canvas.paste(_c36, (576, 2804), _c36)
except Exception:
    pass
layout["9_2273_creator_followers"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/37_text_5.34.png
try:
    _c37 = get_crop(37, 95, 49)
    canvas.paste(_c37, (20, 13), _c37)
except Exception:
    pass
layout["5.34"] = [20, 13, 115, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/38_text_More_events_you_II_love.png
try:
    _c38 = get_crop(38, 1344, 396)
    canvas.paste(_c38, (48, 490), _c38)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/39_text_Thu_May_2.png
try:
    _c39 = get_crop(39, 195, 48)
    canvas.paste(_c39, (389, 2522), _c39)
except Exception:
    pass
layout["Thu,_May_2"] = [389, 2522, 584, 2570]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/40_text_6_00_PM_EDT.png
try:
    _c40 = get_crop(40, 1344, 346)
    canvas.paste(_c40, (48, 2470), _c40)
except Exception:
    pass
layout["6:00_PM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/41_text_Free.png
try:
    _c41 = get_crop(41, 78, 38)
    canvas.paste(_c41, (274, 2561), _c41)
except Exception:
    pass
layout["Free"] = [274, 2561, 352, 2599]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/42_text_How_to_Break_Into_Tech_Learn_to_Code_wit.png
try:
    _c42 = get_crop(42, 1344, 346)
    canvas.paste(_c42, (48, 2470), _c42)
except Exception:
    pass
layout["How_to_Break_Into_Tech:_L"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_01_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-3/43_clickable_More.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (1152, 2804), _c43)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
