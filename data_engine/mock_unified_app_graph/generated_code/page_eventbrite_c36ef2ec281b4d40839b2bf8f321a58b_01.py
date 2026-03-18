# page_id: page_eventbrite_c36ef2ec281b4d40839b2bf8f321a58b_01
# screenshot: 2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3.png
# step_index: 1/8
# task: Open Eventbrite. Set the city to "Chicago". Select the "Fashion" category and view the recommended events. See the date of the first play and its venue.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structure drawing for the mobile UI page
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
bg_color = "#fbfbfd"         # overall page background (slightly off-white)
status_bar_color = "#bfbfbf" # top status bar grey
toolbar_bg = "#ffffff"       # header/toolbar background
divider_color = "#e9e9ef"    # subtle dividers
card_shadow = "#e9eaf2"      # soft shadow under cards
card_bg = "#ffffff"          # card background
bottom_nav_bg = "#ffffff"    # bottom nav background
thin_div = "#f2f3f6"         # very thin divider between cards

# Fill overall background
draw.rectangle([(0, 0), (1440, 2960)], fill=bg_color)

# Status bar (top area)
status_h = 88
draw.rectangle([(0, 0), (1440, status_h)], fill=status_bar_color)

# Header / toolbar area below status bar
toolbar_top = status_h
toolbar_bottom = status_h + 72
draw.rectangle([(0, toolbar_top), (1440, toolbar_bottom)], fill=toolbar_bg)
# subtle bottom divider under toolbar
draw.line([(24, toolbar_bottom), (1440 - 24, toolbar_bottom)], fill=divider_color, width=2)

# Main content area start (leave space for large "More events you'll love" heading)
content_top = toolbar_bottom + 36

# Draw repeating event card backgrounds (rounded rectangles with subtle shadow)
card_x0 = 48
card_x1 = card_x0 + 1344  # matches detected width
card_height = 220         # approximate card height (background for each list item)
card_spacing = 36
num_cards = 7

for i in range(num_cards):
    y0 = content_top + i * (card_height + card_spacing)
    y1 = y0 + card_height

    # shadow (slightly offset)
    draw.rectangle([(card_x0 + 6, y0 + 8), (card_x1 + 6, y1 + 8)], fill=card_shadow)

    # card background (rounded)
    try:
        # use rounded_rectangle if available
        draw.rounded_rectangle([(card_x0, y0), (card_x1, y1)], radius=14, fill=card_bg, outline=thin_div, width=1)
    except Exception:
        # fallback: draw a normal rect if rounded not present
        draw.rectangle([(card_x0, y0), (card_x1, y1)], fill=card_bg, outline=thin_div, width=1)

    # light separator line below each card (gives visual separation)
    sep_y = y1 + int(card_spacing / 2)
    draw.line([(card_x0 + 8, sep_y), (card_x1 - 8, sep_y)], fill=divider_color, width=1)

# Draw a subtle long divider where the "More events you'll love" title would sit (do not draw text)
title_div_y = content_top - 18
draw.line([(48, title_div_y), (1440 - 48, title_div_y)], fill=divider_color, width=1)

# Bottom navigation bar background and top divider
nav_h = 120
nav_top = 2960 - nav_h
draw.rectangle([(0, nav_top), (1440, 2960)], fill=bottom_nav_bg)
# top divider of nav bar
draw.line([(24, nav_top), (1440 - 24, nav_top)], fill=divider_color, width=2)

# Small subtle shadow above bottom nav to separate it from content
draw.line([(24, nav_top + 2), (1440 - 24, nav_top + 2)], fill=card_shadow, width=1)

# Draw faint vertical guide lines for content margins (non-intrusive)
draw.line([(card_x0, toolbar_bottom + 4), (card_x0, nav_top - 8)], fill=thin_div, width=1)
draw.line([(card_x1, toolbar_bottom + 4), (card_x1, nav_top - 8)], fill=thin_div, width=1)

# Final top status area subtle inner divider (to match screenshot's layered bars)
draw.line([(12, status_h - 2), (1440 - 12, status_h - 2)], fill="#d0d0d0", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/00_icon_Chicago.png
try:
    _c0 = get_crop(0, 388, 117)
    canvas.paste(_c0, (526, 2651), _c0)
except Exception:
    pass
layout["Chicago"] = [526, 2651, 914, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/01_icon_CyPo6.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1678), _c1)
except Exception:
    pass
layout["CyPo6"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/02_icon_ripg_-_LeaTG_Atans.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 2074), _c2)
except Exception:
    pass
layout["ripg_-_LeaTG_Atans"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/03_icon_Okstore.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1282), _c3)
except Exception:
    pass
layout["Okstore"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/04_icon_Q_Search_events.png
try:
    _c4 = get_crop(4, 1179, 144)
    canvas.paste(_c4, (195, 93), _c4)
except Exception:
    pass
layout["Q_Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/05_icon_Sat_Oct_19.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 490), _c5)
except Exception:
    pass
layout["Sat,_Oct_19"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/06_icon_Dovetail_Brewery.png
try:
    _c6 = get_crop(6, 144, 139)
    canvas.paste(_c6, (1140, 1935), _c6)
except Exception:
    pass
layout["Dovetail_Brewery"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 123)
    canvas.paste(_c7, (1140, 2347), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/08_icon_Favorite_button.png
try:
    _c8 = get_crop(8, 144, 139)
    canvas.paste(_c8, (1140, 1539), _c8)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1284, 1935), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/10_icon_Overflow_menu_button.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1284, 2347), _c10)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/11_icon_Favorite_button.png
try:
    _c11 = get_crop(11, 144, 139)
    canvas.paste(_c11, (1140, 1143), _c11)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/12_icon_Favorite_button.png
try:
    _c12 = get_crop(12, 144, 125)
    canvas.paste(_c12, (1140, 761), _c12)
except Exception:
    pass
layout["Favorite_button"] = [1140, 761, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/13_icon_7940_Wolcott_Ave_apt_2_Chicago_IL_USA.png
try:
    _c13 = get_crop(13, 1344, 396)
    canvas.paste(_c13, (48, 490), _c13)
except Exception:
    pass
layout["7940_$_Wolcott_Ave_apt_2,"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 125)
    canvas.paste(_c14, (1284, 761), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 761, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/15_icon_Joliet.png
try:
    _c15 = get_crop(15, 288, 156)
    canvas.paste(_c15, (288, 2804), _c15)
except Exception:
    pass
layout["Joliet"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/16_icon_Overflow_menu_button.png
try:
    _c16 = get_crop(16, 144, 139)
    canvas.paste(_c16, (1284, 1539), _c16)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/17_icon_5.12.png
try:
    _c17 = get_crop(17, 105, 100)
    canvas.paste(_c17, (40, 122), _c17)
except Exception:
    pass
layout["5.12"] = [40, 122, 145, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/18_icon_through_thc_chi.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (0, 2804), _c18)
except Exception:
    pass
layout["through_thc_chi"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/19_icon_5.12.png
try:
    _c19 = get_crop(19, 55, 60)
    canvas.paste(_c19, (183, 2), _c19)
except Exception:
    pass
layout["5.12"] = [183, 2, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/20_icon_ON.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 886), _c20)
except Exception:
    pass
layout["ON"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/21_icon_Overflow_menu_button.png
try:
    _c21 = get_crop(21, 144, 139)
    canvas.paste(_c21, (1284, 1143), _c21)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 60, 58)
    canvas.paste(_c22, (312, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [312, 3, 372, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/23_icon_49_creator_followers.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 886), _c23)
except Exception:
    pass
layout["49_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 50, 59)
    canvas.paste(_c24, (248, 2), _c24)
except Exception:
    pass
layout["icon_24"] = [248, 2, 298, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/25_icon_Indie_Bookstore_Day_at_Goblin_Market.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 1282), _c25)
except Exception:
    pass
layout["Indie_Bookstore_Day_at_Go"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 48, 53)
    canvas.paste(_c26, (1321, 7), _c26)
except Exception:
    pass
layout["icon_26"] = [1321, 7, 1369, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/27_icon_Planting_Seeds_bilingual.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 2074), _c27)
except Exception:
    pass
layout["Planting_Seeds_(bilingual"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 59, 58)
    canvas.paste(_c28, (1212, 4), _c28)
except Exception:
    pass
layout["icon_28"] = [1212, 4, 1271, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/29_icon_Q_Search_events.png
try:
    _c29 = get_crop(29, 44, 56)
    canvas.paste(_c29, (385, 6), _c29)
except Exception:
    pass
layout["Q_Search_events"] = [385, 6, 429, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 41, 55)
    canvas.paste(_c30, (1272, 6), _c30)
except Exception:
    pass
layout["icon_30"] = [1272, 6, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/31_icon_73_creator_followers.png
try:
    _c31 = get_crop(31, 1344, 396)
    canvas.paste(_c31, (48, 1678), _c31)
except Exception:
    pass
layout["73_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/32_icon_5.12.png
try:
    _c32 = get_crop(32, 57, 60)
    canvas.paste(_c32, (116, 2), _c32)
except Exception:
    pass
layout["5.12"] = [116, 2, 173, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/33_icon_Self-Love_in_Nature_Releasing_Grief_thro.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 2074), _c33)
except Exception:
    pass
layout["Self-Love_in_Nature:_Rele"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/34_icon_Grief_R.png
try:
    _c34 = get_crop(34, 1344, 346)
    canvas.paste(_c34, (48, 2470), _c34)
except Exception:
    pass
layout["Grief_R"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/35_icon_6_00_PM_CDT.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 1678), _c35)
except Exception:
    pass
layout["6:00_PM_CDT"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/36_icon_Dovetail_Brewery.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 1678), _c36)
except Exception:
    pass
layout["Dovetail_Brewery"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/37_icon_Discover_Your_Path_To_Healing_With_Our_G.png
try:
    _c37 = get_crop(37, 1344, 346)
    canvas.paste(_c37, (48, 2470), _c37)
except Exception:
    pass
layout["Discover_Your_Path_To_Hea"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/38_text_5.12.png
try:
    _c38 = get_crop(38, 89, 43)
    canvas.paste(_c38, (22, 17), _c38)
except Exception:
    pass
layout["5.12"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/39_text_More_events_you_II_love.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 490), _c39)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/40_text_Tue_May_7.png
try:
    _c40 = get_crop(40, 191, 43)
    canvas.paste(_c40, (390, 2525), _c40)
except Exception:
    pass
layout["Tue,_May_7"] = [390, 2525, 581, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/41_text_6_00_PM_CDT.png
try:
    _c41 = get_crop(41, 1344, 346)
    canvas.paste(_c41, (48, 2470), _c41)
except Exception:
    pass
layout["6:00_PM_CDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/42_text_Joliet.png
try:
    _c42 = get_crop(42, 96, 38)
    canvas.paste(_c42, (390, 2723), _c42)
except Exception:
    pass
layout["Joliet"] = [390, 2723, 486, 2761]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/43_clickable_Favorites.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (576, 2804), _c43)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/44_clickable_Tickets.png
try:
    _c44 = get_crop(44, 288, 156)
    canvas.paste(_c44, (864, 2804), _c44)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_01_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-3/45_clickable_More.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (1152, 2804), _c45)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
