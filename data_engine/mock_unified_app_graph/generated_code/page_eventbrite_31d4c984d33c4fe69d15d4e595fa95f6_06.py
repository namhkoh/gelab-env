# page_id: page_eventbrite_31d4c984d33c4fe69d15d4e595fa95f6_06
# screenshot: 2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8.png
# step_index: 6/14
# task: Open Eventbrite. Look for 'community events' in 'Chicago'. Select the first event happening tomorrow that is not promoted. Check if they have an option for 'refund policy'.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural layout for Event list UI (PIL drawing)
# Uses provided variables: canvas (PIL Image), draw (PIL ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
bg_color = (250, 250, 250)            # overall page background (very light)
status_bar_color = (189, 189, 189)    # top status bar gray
header_bg = (255, 255, 255)           # header/toolbar background (white)
card_bg = (255, 255, 255)             # card background (white)
card_outline = (230, 230, 234)        # subtle card border
separator_color = (241, 241, 244)     # light separators between rows
bottom_nav_bg = (250, 250, 250)       # bottom navigation bar bg
top_divider = (220, 220, 224)         # dividers / subtle lines
shadow_color = (0, 0, 0, 18)          # very light drop shadow (if used)

w, h = canvas.size

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar (top area ~64px tall)
status_h = 64
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)

# Header area behind search + slight padding (do not draw the search field itself)
header_top = status_h
header_bottom = 280  # extends a bit to create a distinct header region
draw.rectangle([(0, header_top), (w, header_bottom)], fill=header_bg)

# Subtle bottom divider under header
draw.line([(48, header_bottom), (w-48, header_bottom)], fill=top_divider, width=1)

# Card/list area: draw rounded card backgrounds for each event group.
# Positions chosen to align with detected groups (y origins from detected elements).
card_x = 48
card_w = 1344
card_corner = 14
card_height = 200  # visual card height (background only)
card_positions_y = [480, 880, 1280, 1680, 2080, 2480]  # approximate top y for each card

for y in card_positions_y:
    x1 = card_x
    y1 = y
    x2 = card_x + card_w
    y2 = y + card_height

    # subtle shadow: thin translucent strip below card (simulated by a faint line)
    shadow_y = y2 + 2
    draw.line([(x1+8, shadow_y), (x2-8, shadow_y)], fill=(230,230,235), width=1)

    # card background with subtle border
    try:
        draw.rounded_rectangle([(x1, y1), (x2, y2)], radius=card_corner, fill=card_bg, outline=card_outline, width=1)
    except Exception:
        # fallback for PIL versions without rounded_rectangle
        draw.rectangle([(x1, y1), (x2, y2)], fill=card_bg, outline=card_outline, width=1)

    # light separator line under each card (space between cards)
    sep_y = y2 + 18
    draw.line([(x1, sep_y), (x2, sep_y)], fill=separator_color, width=1)

# Additional thin separators for smaller grouping near top content
sep_positions = [380, 460, 760, 1160, 1560, 1960, 2360]
for sy in sep_positions:
    draw.line([(48, sy), (w-48, sy)], fill=separator_color, width=1)

# Bottom navigation bar background and top divider (do not draw icons)
bottom_nav_top = 2804
draw.rectangle([(0, bottom_nav_top), (w, h)], fill=bottom_nav_bg)
draw.line([(0, bottom_nav_top), (w, bottom_nav_top)], fill=top_divider, width=1)

# Subtle top-left and top-right safe margins (visual guides)
draw.line([(48, header_top+8), (48, header_bottom-8)], fill=(255,255,255,0), width=0)  # no-op to preserve spacing

# Decorative subtle left column guide (very faint) to indicate content column (non-intrusive)
draw.line([(48, header_bottom+8), (48, h-bottom_nav_top//6)], fill=(245,245,247), width=1)

# End of drawing structural elements. Content (icons/text/images) will be pasted on top.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/00_icon_ripg_-_LeaTG_Atans.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2074), _c0)
except Exception:
    pass
layout["ripg_-_LeaTG_Atans"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/01_icon_EYPCG.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1678), _c1)
except Exception:
    pass
layout["EYPCG"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/02_icon_Chicago.png
try:
    _c2 = get_crop(2, 388, 117)
    canvas.paste(_c2, (526, 2651), _c2)
except Exception:
    pass
layout["Chicago"] = [526, 2651, 914, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/03_icon_iokstore.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1282), _c3)
except Exception:
    pass
layout["iokstore"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/04_icon_Search_events.png
try:
    _c4 = get_crop(4, 1179, 144)
    canvas.paste(_c4, (195, 93), _c4)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/05_icon_Sat_Oct_19.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 490), _c5)
except Exception:
    pass
layout["Sat,_Oct_19"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/06_icon_Dovetail_Brewery.png
try:
    _c6 = get_crop(6, 144, 139)
    canvas.paste(_c6, (1140, 1935), _c6)
except Exception:
    pass
layout["Dovetail_Brewery"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 123)
    canvas.paste(_c7, (1140, 2347), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/08_icon_Favorite_button.png
try:
    _c8 = get_crop(8, 144, 139)
    canvas.paste(_c8, (1140, 1539), _c8)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 49, 65)
    canvas.paste(_c9, (1153, 2), _c9)
except Exception:
    pass
layout["icon_9"] = [1153, 2, 1202, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/10_icon_7940_Wolcott_Ave_apt_2_Chicago_IL_USA.png
try:
    _c10 = get_crop(10, 1344, 396)
    canvas.paste(_c10, (48, 490), _c10)
except Exception:
    pass
layout["7940_$_Wolcott_Ave_apt_2,"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 139)
    canvas.paste(_c11, (1284, 1935), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 123)
    canvas.paste(_c12, (1284, 2347), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/13_icon_Favorite_button.png
try:
    _c13 = get_crop(13, 144, 139)
    canvas.paste(_c13, (1140, 1143), _c13)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/14_icon_Favorite_button.png
try:
    _c14 = get_crop(14, 144, 125)
    canvas.paste(_c14, (1140, 761), _c14)
except Exception:
    pass
layout["Favorite_button"] = [1140, 761, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/15_icon_Joliet.png
try:
    _c15 = get_crop(15, 288, 156)
    canvas.paste(_c15, (288, 2804), _c15)
except Exception:
    pass
layout["Joliet"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/16_icon_Overflow_menu_button.png
try:
    _c16 = get_crop(16, 144, 125)
    canvas.paste(_c16, (1284, 761), _c16)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 761, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/17_icon_Overflow_menu_button.png
try:
    _c17 = get_crop(17, 144, 139)
    canvas.paste(_c17, (1284, 1539), _c17)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/18_icon_47_creator_followers.png
try:
    _c18 = get_crop(18, 1344, 396)
    canvas.paste(_c18, (48, 886), _c18)
except Exception:
    pass
layout["47_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/19_icon_8.07.png
try:
    _c19 = get_crop(19, 103, 100)
    canvas.paste(_c19, (41, 122), _c19)
except Exception:
    pass
layout["8.07"] = [41, 122, 144, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/20_icon_through_thc_chi.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (0, 2804), _c20)
except Exception:
    pass
layout["through_thc_chi"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/21_icon_8.07.png
try:
    _c21 = get_crop(21, 55, 60)
    canvas.paste(_c21, (183, 2), _c21)
except Exception:
    pass
layout["8.07"] = [183, 2, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/22_icon_Overflow_menu_button.png
try:
    _c22 = get_crop(22, 144, 139)
    canvas.paste(_c22, (1284, 1143), _c22)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 59, 57)
    canvas.paste(_c23, (312, 4), _c23)
except Exception:
    pass
layout["icon_23"] = [312, 4, 371, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/24_icon_ON.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 886), _c24)
except Exception:
    pass
layout["ON"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 97, 59)
    canvas.paste(_c25, (1216, 4), _c25)
except Exception:
    pass
layout["icon_25"] = [1216, 4, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/26_icon_Planting_Seeds_bilingual.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 2074), _c26)
except Exception:
    pass
layout["Planting_Seeds_(bilingual"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 50, 58)
    canvas.paste(_c27, (248, 3), _c27)
except Exception:
    pass
layout["icon_27"] = [248, 3, 298, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 48, 53)
    canvas.paste(_c28, (1321, 7), _c28)
except Exception:
    pass
layout["icon_28"] = [1321, 7, 1369, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/29_icon_8.07.png
try:
    _c29 = get_crop(29, 59, 59)
    canvas.paste(_c29, (114, 3), _c29)
except Exception:
    pass
layout["8.07"] = [114, 3, 173, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/30_icon_Indie_Bookstore_Day_at_Goblin_Market.png
try:
    _c30 = get_crop(30, 1344, 396)
    canvas.paste(_c30, (48, 1282), _c30)
except Exception:
    pass
layout["Indie_Bookstore_Day_at_Go"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/31_icon_icon_31.png
try:
    _c31 = get_crop(31, 44, 55)
    canvas.paste(_c31, (385, 7), _c31)
except Exception:
    pass
layout["icon_31"] = [385, 7, 429, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/32_icon_Grief_R.png
try:
    _c32 = get_crop(32, 1344, 346)
    canvas.paste(_c32, (48, 2470), _c32)
except Exception:
    pass
layout["Grief_R"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/33_icon_Self-Love_in_Nature_Releasing_Grief_thro.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 2074), _c33)
except Exception:
    pass
layout["Self-Love_in_Nature:_Rele"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/34_icon_6_00_PM_CDT.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 1678), _c34)
except Exception:
    pass
layout["6:00_PM_CDT"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/35_icon_6_00_PM_CDT.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 1678), _c35)
except Exception:
    pass
layout["6:00_PM_CDT"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/36_icon_Discover_Your_Path_To_Healing_With_Our_G.png
try:
    _c36 = get_crop(36, 1344, 346)
    canvas.paste(_c36, (48, 2470), _c36)
except Exception:
    pass
layout["Discover_Your_Path_To_Hea"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/37_icon_Dovetail_Brewery.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 1678), _c37)
except Exception:
    pass
layout["Dovetail_Brewery"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/38_text_8.07.png
try:
    _c38 = get_crop(38, 89, 43)
    canvas.paste(_c38, (20, 17), _c38)
except Exception:
    pass
layout["8.07"] = [20, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/39_text_More_events_you_II_love.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 490), _c39)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/40_text_Tue_Apr_23.png
try:
    _c40 = get_crop(40, 200, 43)
    canvas.paste(_c40, (390, 2525), _c40)
except Exception:
    pass
layout["Tue,_Apr_23"] = [390, 2525, 590, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/41_text_7_00_PM_CDT.png
try:
    _c41 = get_crop(41, 1344, 346)
    canvas.paste(_c41, (48, 2470), _c41)
except Exception:
    pass
layout["7:00_PM_CDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/42_text_Joliet.png
try:
    _c42 = get_crop(42, 96, 38)
    canvas.paste(_c42, (390, 2723), _c42)
except Exception:
    pass
layout["Joliet"] = [390, 2723, 486, 2761]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/43_clickable_Favorites.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (576, 2804), _c43)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/44_clickable_Tickets.png
try:
    _c44 = get_crop(44, 288, 156)
    canvas.paste(_c44, (864, 2804), _c44)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_06_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-8/45_clickable_More.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (1152, 2804), _c45)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
