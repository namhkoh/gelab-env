# page_id: page_eventbrite_d7ac75f457a4487c904e7baa93180729_03
# screenshot: 2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5.png
# step_index: 3/11
# task: Open Eventbrite. Search for 'Cooking' classes. Filter to only show free events that occur in the weekend. Select the first event and proceed to checkout.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw background and structural UI elements for the mobile page
# available variables: canvas (PIL Image 1440x2960 RGB), draw (ImageDraw), fonts: font_sm,font_md,font_lg,font_xl

W, H = canvas.size

# Colors
bg_color = "#FFFFFF"            # dominant background
status_bar_color = "#CFCFCF"    # top status bar gray
header_bg = "#FFFFFF"           # search/header bg (white)
search_underline = "#1E5BFF"    # vivid blue underline under search
divider_color = "#E6E6E6"       # light divider lines
card_outline = "#EBEBEB"        # card border
card_bg = "#FFFFFF"             # card background (keeps white)
nav_bar_bg = "#FFFFFF"          # bottom nav background

# Clear/fill full background (canvas already white, but ensure)
draw.rectangle([(0,0),(W,H)], fill=bg_color)

# Status bar (top area)
status_h = 72
draw.rectangle([(0,0),(W,status_h)], fill=status_bar_color)

# Header / Search area (below status bar)
header_y0 = status_h
header_y1 = 152
draw.rectangle([(0,header_y0),(W,header_y1)], fill=header_bg)

# Blue underline for the search field
underline_x0 = 48
underline_x1 = W - 48
underline_y = 146
draw.line([(underline_x0, underline_y), (underline_x1, underline_y)], fill=search_underline, width=6)

# Thin divider under header
draw.line([(24, header_y1+2), (W-24, header_y1+2)], fill=divider_color, width=1)

# Section separators (e.g., under "Popular" list area)
# approximate positions based on screenshot layout
popular_div_y = 360
draw.line([(24, popular_div_y), (W-24, popular_div_y)], fill=divider_color, width=1)

events_div_y = 1020
draw.line([(24, events_div_y), (W-24, events_div_y)], fill=divider_color, width=1)

# Event list card backgrounds
card_x0 = 48
card_x1 = 48 + 1344
card_h = 396
card_radius = 12

# y-positions for event cards (from detected elements)
card_ys = [1117, 1513, 1909, 2305, 2804]
for y in card_ys:
    y0 = y
    y1 = y + card_h
    # subtle card shadow: a very light thin line below card to suggest separation
    shadow_y = y1 + 6
    draw.line([(card_x0+8, shadow_y), (card_x1-8, shadow_y)], fill="#F5F5F5", width=8)
    # rounded white card with light border
    try:
        draw.rounded_rectangle([(card_x0, y0), (card_x1, y1)], radius=card_radius, fill=card_bg, outline=card_outline, width=1)
    except AttributeError:
        # fallback if rounded_rectangle not available
        draw.rectangle([(card_x0, y0), (card_x1, y1)], fill=card_bg, outline=card_outline, width=1)

    # thin separator line under each card
    draw.line([(card_x0+12, y1+1), (card_x1-12, y1+1)], fill=divider_color, width=1)

# Bottom navigation bar background and top divider
nav_h = 120
nav_y0 = H - nav_h
draw.rectangle([(0, nav_y0), (W, H)], fill=nav_bar_bg)
draw.line([(24, nav_y0), (W-24, nav_y0)], fill=divider_color, width=1)

# Subtle left content padding guide (very faint vertical rule to define content column)
# (This is purely structural and extremely light so it won't conflict with pasted content)
draw.line([(48, header_y1+8), (48, H - nav_h - 8)], fill="#FAFAFA", width=2)
draw.line([(W-48, header_y1+8), (W-48, H - nav_h - 8)], fill="#FAFAFA", width=2)

# Additional horizontal separators for list rhythm (light)
for y in range(420, events_div_y, 120):
    draw.line([(48, y), (W-48, y)], fill="#FBFBFB", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/00_icon_Events.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1117), _c0)
except Exception:
    pass
layout["Events"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/01_icon_Class.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 2305), _c1)
except Exception:
    pass
layout["Class"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/02_icon_Cooking.png
try:
    _c2 = get_crop(2, 1344, 191)
    canvas.paste(_c2, (48, 72), _c2)
except Exception:
    pass
layout["Cooking]"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/03_icon_WhhoIL_VAGAVADA_ANIAVOlC.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1909), _c3)
except Exception:
    pass
layout["WhhoIL_VAGAVADA_ANIAVOlC"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 41, 51)
    canvas.paste(_c4, (254, 9), _c4)
except Exception:
    pass
layout["icon_4"] = [254, 9, 295, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/05_icon_Cooking.png
try:
    _c5 = get_crop(5, 54, 56)
    canvas.paste(_c5, (314, 6), _c5)
except Exception:
    pass
layout["Cooking]"] = [314, 6, 368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 50, 56)
    canvas.paste(_c6, (185, 5), _c6)
except Exception:
    pass
layout["icon_6"] = [185, 5, 235, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/07_icon_4.38.png
try:
    _c7 = get_crop(7, 54, 57)
    canvas.paste(_c7, (116, 5), _c7)
except Exception:
    pass
layout["4.38"] = [116, 5, 170, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/08_icon_22_creator_followers.png
try:
    _c8 = get_crop(8, 1344, 396)
    canvas.paste(_c8, (48, 1513), _c8)
except Exception:
    pass
layout["22_creator_followers"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/09_icon_Tickets.png
try:
    _c9 = get_crop(9, 288, 156)
    canvas.paste(_c9, (864, 2804), _c9)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/10_icon_4.38.png
try:
    _c10 = get_crop(10, 105, 101)
    canvas.paste(_c10, (64, 120), _c10)
except Exception:
    pass
layout["4.38"] = [64, 120, 169, 221]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/11_icon_16_4_00_PM_PDT.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (576, 2804), _c11)
except Exception:
    pass
layout["16_'_4:00_PM_PDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/12_icon_Cancel.png
try:
    _c12 = get_crop(12, 45, 57)
    canvas.paste(_c12, (1323, 4), _c12)
except Exception:
    pass
layout["Cancel"] = [1323, 4, 1368, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/13_icon_Thu.png
try:
    _c13 = get_crop(13, 288, 156)
    canvas.paste(_c13, (288, 2804), _c13)
except Exception:
    pass
layout["Thu,"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/14_icon_9_30_AM_PDT.png
try:
    _c14 = get_crop(14, 1344, 396)
    canvas.paste(_c14, (48, 1513), _c14)
except Exception:
    pass
layout["9:30_AM_PDT"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/15_icon_Vegan_Brazilian_Cooking_Experience.png
try:
    _c15 = get_crop(15, 1344, 396)
    canvas.paste(_c15, (48, 2305), _c15)
except Exception:
    pass
layout["Vegan_Brazilian_Cooking_E"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/16_icon_More.png
try:
    _c16 = get_crop(16, 288, 156)
    canvas.paste(_c16, (1152, 2804), _c16)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/17_icon_cooking_classes.png
try:
    _c17 = get_crop(17, 1344, 120)
    canvas.paste(_c17, (48, 378), _c17)
except Exception:
    pass
layout["cooking_classes"] = [48, 378, 1392, 498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/18_icon_4.38.png
try:
    _c18 = get_crop(18, 91, 59)
    canvas.paste(_c18, (15, 3), _c18)
except Exception:
    pass
layout["4.38"] = [15, 3, 106, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/19_icon_Cancel.png
try:
    _c19 = get_crop(19, 84, 61)
    canvas.paste(_c19, (1215, 1), _c19)
except Exception:
    pass
layout["Cancel"] = [1215, 1, 1299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/20_icon_Cooking_Party.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 1513), _c20)
except Exception:
    pass
layout["Cooking_Party"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/21_icon_Cooking_Class_Experience.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 1117), _c21)
except Exception:
    pass
layout["Cooking_Class_Experience"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/22_icon_Cancel.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1099, 96), _c22)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/23_icon_Cancel.png
try:
    _c23 = get_crop(23, 149, 144)
    canvas.paste(_c23, (1243, 97), _c23)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/24_icon_COOKING_THR.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 1513), _c24)
except Exception:
    pass
layout["COOKING_THR"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/25_icon_Cancel.png
try:
    _c25 = get_crop(25, 42, 59)
    canvas.paste(_c25, (1272, 3), _c25)
except Exception:
    pass
layout["Cancel"] = [1272, 3, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/26_icon_Cooking_through_a_Cultural_Lens.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 1909), _c26)
except Exception:
    pass
layout["Cooking_through_a_Cultura"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/27_icon_Events.png
try:
    _c27 = get_crop(27, 86, 84)
    canvas.paste(_c27, (37, 892), _c27)
except Exception:
    pass
layout["Events"] = [37, 892, 123, 976]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 87, 92)
    canvas.paste(_c28, (39, 768), _c28)
except Exception:
    pass
layout["icon_28"] = [39, 768, 126, 860]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/29_icon_mmer.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (0, 2804), _c29)
except Exception:
    pass
layout["mmer"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/30_icon_8_54creator_followers.png
try:
    _c30 = get_crop(30, 1344, 396)
    canvas.paste(_c30, (48, 1909), _c30)
except Exception:
    pass
layout["8_54creator_followers"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/31_icon_3_00_PM_PDT.png
try:
    _c31 = get_crop(31, 1344, 396)
    canvas.paste(_c31, (48, 1909), _c31)
except Exception:
    pass
layout["3:00_PM_PDT"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/32_text_Popular.png
try:
    _c32 = get_crop(32, 224, 78)
    canvas.paste(_c32, (41, 298), _c32)
except Exception:
    pass
layout["Popular"] = [41, 298, 265, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/33_text_couples_cooking_class.png
try:
    _c33 = get_crop(33, 1344, 120)
    canvas.paste(_c33, (48, 498), _c33)
except Exception:
    pass
layout["couples_cooking_class"] = [48, 498, 1392, 618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/34_text_cooking_class.png
try:
    _c34 = get_crop(34, 1344, 120)
    canvas.paste(_c34, (48, 618), _c34)
except Exception:
    pass
layout["cooking_class"] = [48, 618, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/35_text_vegan_cooking_class.png
try:
    _c35 = get_crop(35, 1344, 120)
    canvas.paste(_c35, (48, 738), _c35)
except Exception:
    pass
layout["vegan_cooking_class"] = [48, 738, 1392, 858]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/36_text_gluten_free_cooking_class.png
try:
    _c36 = get_crop(36, 1344, 144)
    canvas.paste(_c36, (48, 858), _c36)
except Exception:
    pass
layout["gluten_free_cooking_class"] = [48, 858, 1392, 1002]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/37_text_Events.png
try:
    _c37 = get_crop(37, 191, 61)
    canvas.paste(_c37, (45, 1026), _c37)
except Exception:
    pass
layout["Events"] = [45, 1026, 236, 1087]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/38_text_Fri.png
try:
    _c38 = get_crop(38, 68, 48)
    canvas.paste(_c38, (389, 2392), _c38)
except Exception:
    pass
layout["Fri,"] = [389, 2392, 457, 2440]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/39_text_10.png
try:
    _c39 = get_crop(39, 55, 38)
    canvas.paste(_c39, (528, 2395), _c39)
except Exception:
    pass
layout["10"] = [528, 2395, 583, 2433]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/40_text_6_00_PM_PDT.png
try:
    _c40 = get_crop(40, 1344, 396)
    canvas.paste(_c40, (48, 2305), _c40)
except Exception:
    pass
layout["6:00_PM_PDT"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/41_text_93_Windward.png
try:
    _c41 = get_crop(41, 219, 41)
    canvas.paste(_c41, (392, 2517), _c41)
except Exception:
    pass
layout["93_Windward"] = [392, 2517, 611, 2558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/42_text_8_11_creator_followers.png
try:
    _c42 = get_crop(42, 1344, 396)
    canvas.paste(_c42, (48, 2305), _c42)
except Exception:
    pass
layout["8_11_creator_followers"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/43_text_mmer.png
try:
    _c43 = get_crop(43, 71, 25)
    canvas.paste(_c43, (44, 2782), _c43)
except Exception:
    pass
layout["mmer"] = [44, 2782, 115, 2807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/44_text_Thu.png
try:
    _c44 = get_crop(44, 86, 45)
    canvas.paste(_c44, (390, 2757), _c44)
except Exception:
    pass
layout["Thu,"] = [390, 2757, 476, 2802]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_03_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-5/45_text_16_4_00_PM_PDT.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (576, 2804), _c45)
except Exception:
    pass
layout["16_'_4:00_PM_PDT"] = [576, 2804, 864, 2960]
