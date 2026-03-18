# page_id: page_eventbrite_86c0bd1901f44c94916665f4058f9b6d_01
# screenshot: 2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3.png
# step_index: 1/11
# task: Open Eventbrite. Set the city to Los Angeles. Select the 'Food & Drink' category. What's the date of the first event that is not promoted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background with a soft off-white to match the app's page background
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFAFC")

# Status bar (top area) - light gray band for time/signal background
status_bar_height = 80
draw.rectangle([(0, 0), (1440, status_bar_height)], fill="#CFCFCF")

# Subtle bottom divider under status bar
draw.line([(0, status_bar_height), (1440, status_bar_height)], fill="#BFBFBF", width=1)

# Header / toolbar background area (below status bar). Keep it slightly warmer white to differentiate.
header_top = status_bar_height
header_bottom = 240
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")
# Thin divider line under header
draw.line([(48, header_bottom), (1440-48, header_bottom)], fill="#E8E6EA", width=1)

# Main content area background (slight tint distinct from pure white)
content_top = header_bottom + 24
draw.rectangle([(0, content_top), (1440, 2960 - 160)], fill="#FBFAFC")

# Define the card positions (left aligned at x=48, width 1344, height 396), matching event rows.
card_x = 48
card_w = 1344
card_h = 396
card_radius = 18

# Row Y positions inferred from detected elements in the screenshot
row_ys = [490, 886, 1282, 1678, 2074, 2347]  # these are the top y for each event card

# Draw subtle rounded card backgrounds with very light shadow lines
for y in row_ys:
    x0 = card_x
    y0 = y
    x1 = card_x + card_w
    y1 = y + card_h

    # Card background (white)
    draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=card_radius, fill="#FFFFFF")

    # Light inner top highlight
    highlight_y = y0 + 6
    draw.rectangle([(x0 + 2, y0 + 2), (x1 - 2, highlight_y)], fill="#FFFFFF", outline=None)

    # Subtle drop shadow (thin line under card)
    shadow_y = y1 + 6
    draw.line([(x0 + 6, shadow_y), (x1 - 6, shadow_y)], fill="#EFEFF1", width=2)
    # Slight darker shadow closer to the card bottom for depth
    draw.line([(x0 + 6, y1 + 2), (x1 - 6, y1 + 2)], fill="#F4F4F6", width=1)

    # Inner left image placeholder background area (do not draw any image content).
    # Draw a muted rectangle where thumbnails will be pasted (keeps layout but not duplicating actual image content).
    thumb_w = int(card_h * 0.8)  # approximate square-ish thumbnail
    thumb_margin_v = (card_h - thumb_w) // 2
    thumb_x0 = x0 + 10
    thumb_y0 = y0 + thumb_margin_v
    thumb_x1 = thumb_x0 + thumb_w
    thumb_y1 = thumb_y0 + thumb_w
    # Use a very subtle soft-gray background for thumbnail area to indicate image panel, not the image itself.
    draw.rounded_rectangle([(thumb_x0, thumb_y0), (thumb_x1, thumb_y1)], radius=8, fill="#F6F6F8", outline="#EAEAF0")

    # Vertical separator line (between thumbnail and text area)
    sep_x = thumb_x1 + 18
    draw.line([(sep_x, y0 + 18), (sep_x, y1 - 18)], fill="#F0EEF3", width=1)

# Draw separators between content sections (above card list and between groups)
# a divider just above the first card
draw.line([(48, row_ys[0] - 24), (1392, row_ys[0] - 24)], fill="#EFEAF0", width=1)

# Additional subtle horizontal separators between each card group
for y in row_ys:
    sep_y = y + card_h + 12
    draw.line([(48, sep_y), (1392, sep_y)], fill="#F3F2F5", width=1)

# Bottom navigation bar area (reserve space and draw top divider)
bottom_nav_top = 2804
draw.rectangle([(0, bottom_nav_top), (1440, 2960)], fill="#FFFFFF")
draw.line([(0, bottom_nav_top), (1440, bottom_nav_top)], fill="#E6E4E8", width=1)

# Floating city selector notch area: draw a subtle shadow/backdrop behind where the UI will paste the selector,
# but avoid drawing any icons or text. Place a very soft blur-like rectangle (non-intrusive).
selector_center_x = 720
selector_center_y = 2651
selector_w = 420
selector_h = 86
selector_box = [(selector_center_x - selector_w//2, selector_center_y - selector_h//2),
                (selector_center_x + selector_w//2, selector_center_y + selector_h//2)]
draw.rounded_rectangle(selector_box, radius=44, fill="#FFFFFF", outline="#EDEAF0")
# Soft top and bottom hairlines to give depth
draw.line([(selector_box[0][0]+2, selector_box[0][1]+1), (selector_box[1][0]-2, selector_box[0][1]+1)], fill="#FBF9FB", width=1)
draw.line([(selector_box[0][0]+2, selector_box[1][1]-1), (selector_box[1][0]-2, selector_box[1][1]-1)], fill="#F2F1F4", width=1)

# Final subtle vignette lines to anchor the main content area
draw.line([(48, content_top), (1392, content_top)], fill="#F0EEF3", width=1)
draw.line([(48, bottom_nav_top - 16), (1392, bottom_nav_top - 16)], fill="#F4F3F6", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/00_icon_Grief_R.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2074), _c0)
except Exception:
    pass
layout["Grief_R"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/01_icon_Pc.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1678), _c1)
except Exception:
    pass
layout["Pc"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/02_icon_EYPOG.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 490), _c2)
except Exception:
    pass
layout["EYPOG"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/03_icon_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 123)
    canvas.paste(_c4, (1140, 1951), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 123)
    canvas.paste(_c5, (1140, 1555), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1555, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/06_icon_Recovery_Roadmap.png
try:
    _c6 = get_crop(6, 1344, 396)
    canvas.paste(_c6, (48, 1678), _c6)
except Exception:
    pass
layout["Recovery_Roadmap"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/07_icon_Chicago.png
try:
    _c7 = get_crop(7, 388, 117)
    canvas.paste(_c7, (526, 2651), _c7)
except Exception:
    pass
layout["Chicago"] = [526, 2651, 914, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 123)
    canvas.paste(_c8, (1284, 1555), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1555, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/09_icon_47_creat..png
try:
    _c9 = get_crop(9, 288, 156)
    canvas.paste(_c9, (288, 2804), _c9)
except Exception:
    pass
layout["47_creat."] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/10_icon_Favorite_button.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1140, 2347), _c10)
except Exception:
    pass
layout["Favorite_button"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/11_icon_Favorite_button.png
try:
    _c11 = get_crop(11, 144, 125)
    canvas.paste(_c11, (1140, 1157), _c11)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1157, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/12_icon_Planting_Seeds_bilingual.png
try:
    _c12 = get_crop(12, 1344, 396)
    canvas.paste(_c12, (48, 1282), _c12)
except Exception:
    pass
layout["Planting_Seeds_(bilingual"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 123)
    canvas.paste(_c13, (1284, 1951), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/14_icon_Light_olloving_Kindret.png
try:
    _c14 = get_crop(14, 1344, 396)
    canvas.paste(_c14, (48, 1282), _c14)
except Exception:
    pass
layout["Light_olloving_Kindret"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/15_icon_Favorite_button.png
try:
    _c15 = get_crop(15, 144, 139)
    canvas.paste(_c15, (1140, 747), _c15)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/16_icon_Overflow_menu_button.png
try:
    _c16 = get_crop(16, 144, 123)
    canvas.paste(_c16, (1284, 2347), _c16)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/17_icon_Overflow_menu_button.png
try:
    _c17 = get_crop(17, 144, 125)
    canvas.paste(_c17, (1284, 1157), _c17)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1157, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/18_icon_WizaRd_staFFING.png
try:
    _c18 = get_crop(18, 1344, 396)
    canvas.paste(_c18, (48, 886), _c18)
except Exception:
    pass
layout["WizaRd_staFFING"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/19_icon_7.13.png
try:
    _c19 = get_crop(19, 55, 60)
    canvas.paste(_c19, (183, 2), _c19)
except Exception:
    pass
layout["7.13"] = [183, 2, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/20_icon_Overflow_menu_button.png
try:
    _c20 = get_crop(20, 144, 139)
    canvas.paste(_c20, (1284, 747), _c20)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/21_icon_7.13.png
try:
    _c21 = get_crop(21, 100, 97)
    canvas.paste(_c21, (42, 123), _c21)
except Exception:
    pass
layout["7.13"] = [42, 123, 142, 220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 59, 57)
    canvas.paste(_c22, (312, 4), _c22)
except Exception:
    pass
layout["icon_22"] = [312, 4, 371, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/23_icon_Home.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (0, 2804), _c23)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 50, 58)
    canvas.paste(_c24, (248, 3), _c24)
except Exception:
    pass
layout["icon_24"] = [248, 3, 298, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 47, 51)
    canvas.paste(_c25, (1321, 8), _c25)
except Exception:
    pass
layout["icon_25"] = [1321, 8, 1368, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/26_icon_7.13.png
try:
    _c26 = get_crop(26, 58, 61)
    canvas.paste(_c26, (115, 2), _c26)
except Exception:
    pass
layout["7.13"] = [115, 2, 173, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/27_icon_7940_Wolcott_Ave_apt_2_Chicago_IL_USA.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 886), _c27)
except Exception:
    pass
layout["7940_$_Wolcott_Ave_apt_2,"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 55, 56)
    canvas.paste(_c28, (1213, 6), _c28)
except Exception:
    pass
layout["icon_28"] = [1213, 6, 1268, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/29_icon_Recovery_Roadmap.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 2074), _c29)
except Exception:
    pass
layout["Recovery_Roadmap"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 41, 54)
    canvas.paste(_c30, (1272, 7), _c30)
except Exception:
    pass
layout["icon_30"] = [1272, 7, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/31_icon_icon_31.png
try:
    _c31 = get_crop(31, 44, 55)
    canvas.paste(_c31, (385, 7), _c31)
except Exception:
    pass
layout["icon_31"] = [385, 7, 429, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/32_icon_Indie_Sound_Carnival.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 886), _c32)
except Exception:
    pass
layout["Indie_Sound_Carnival"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/33_icon_Eggers_Grove.png
try:
    _c33 = get_crop(33, 223, 54)
    canvas.paste(_c33, (391, 1526), _c33)
except Exception:
    pass
layout["Eggers_Grove"] = [391, 1526, 614, 1580]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/34_icon_72_creator_followers.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 490), _c34)
except Exception:
    pass
layout["72_creator_followers"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/35_icon_Discover_Your_Path_To_Healing_With_Our_G.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 2074), _c35)
except Exception:
    pass
layout["Discover_Your_Path_To_Hea"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/36_icon_rJ_U_5I0.png
try:
    _c36 = get_crop(36, 288, 156)
    canvas.paste(_c36, (576, 2804), _c36)
except Exception:
    pass
layout["rJ'U'5I0"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/37_text_7.13.png
try:
    _c37 = get_crop(37, 91, 43)
    canvas.paste(_c37, (20, 15), _c37)
except Exception:
    pass
layout["7.13"] = [20, 15, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/38_text_More_events_you_II_love.png
try:
    _c38 = get_crop(38, 1344, 396)
    canvas.paste(_c38, (48, 490), _c38)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/39_text_Sat_May_18.png
try:
    _c39 = get_crop(39, 211, 48)
    canvas.paste(_c39, (389, 2554), _c39)
except Exception:
    pass
layout["Sat;_May_18"] = [389, 2554, 600, 2602]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/40_text_I_00_PM_CDT.png
try:
    _c40 = get_crop(40, 1344, 346)
    canvas.paste(_c40, (48, 2470), _c40)
except Exception:
    pass
layout["I:00_PM_CDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/41_text_2S_North_Lincoln_Avenue.png
try:
    _c41 = get_crop(41, 288, 156)
    canvas.paste(_c41, (864, 2804), _c41)
except Exception:
    pass
layout["2S,_North_Lincoln_Avenue,"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/42_text_rJ_U_5I0.png
try:
    _c42 = get_crop(42, 388, 117)
    canvas.paste(_c42, (526, 2651), _c42)
except Exception:
    pass
layout["rJ'U'5I0"] = [526, 2651, 914, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_01_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-3/43_clickable_More.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (1152, 2804), _c43)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
