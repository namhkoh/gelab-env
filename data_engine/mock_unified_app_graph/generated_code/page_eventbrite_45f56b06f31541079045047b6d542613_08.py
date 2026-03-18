# page_id: page_eventbrite_45f56b06f31541079045047b6d542613_08
# screenshot: 2024_4_23_19_27_45f56b06f31541079045047b6d542613-10.png
# step_index: 8/21
# task: Open Eventbrite. Search events 'Yoga session' in New York. Filter free events and set date from May 3 to May 6. What is the location of the first promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the mobile UI page.
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
bg_color = (250, 250, 252)         # very light background
status_bar_color = (160, 160, 160) # muted grey status bar
header_bg = (255, 255, 255)        # header background (white)
accent_blue = (43, 92, 255)        # blue accent for underline
card_bg = (255, 255, 255)          # card background (white)
card_shadow = (240, 240, 245)      # subtle shadow / separator
separator = (230, 230, 235)        # thin separators
nav_top_line = (220, 220, 225)     # nav bar top divider

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar (top area) ~50-64px tall
status_h = 64
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)

# Header / search area
header_top = status_h
header_bottom = 156  # approximate bottom of header/search underline
draw.rectangle([(0, header_top), (w, header_bottom)], fill=header_bg)

# Blue underline for search input (thicker accent)
underline_y = header_bottom
draw.line([(48, underline_y), (w-48, underline_y)], fill=accent_blue, width=6)

# Slight shadow under the underline to separate header from content
draw.line([(0, underline_y+6), (w, underline_y+6)], fill=(245,245,248), width=1)

# Content card backgrounds (rounded rectangles) for each listed event row
# Detected event blocks at y = 390, 786, 1182, 1578, 1974 with height 396
card_left = 48
card_right = 48 + 1344  # 1392
card_positions = [390, 786, 1182, 1578, 1974]
card_height = 396
card_radius = 14

for y in card_positions:
    top = y
    bottom = y + card_height
    # subtle card fill (white) on top of page bg
    draw.rounded_rectangle(
        [(card_left, top), (card_right, bottom)],
        radius=card_radius,
        fill=card_bg,
        outline=None
    )
    # subtle separator / shadow line under each card
    sep_y = bottom + 10
    draw.line([(card_left, sep_y), (card_right, sep_y)], fill=card_shadow, width=1)

    # thin divider line between entries (closer to visual style)
    divider_y = bottom + 26
    draw.line([(card_left, divider_y), (card_right, divider_y)], fill=separator, width=1)

# Large subtle section separators occasionally aligned to left margin for structure
# (keeps areas visually separated without drawing any text or icons)
for sep_y in [320, 720, 1116, 1512, 1908]:
    draw.line([(card_left, sep_y), (card_right, sep_y)], fill=(245,245,247), width=1)

# Bottom navigation bar background and top divider
nav_top = 2804
nav_bottom = h
draw.rectangle([(0, nav_top), (w, nav_bottom)], fill=(255,255,255))
# top divider for nav
draw.line([(0, nav_top), (w, nav_top)], fill=nav_top_line, width=2)

# Slight inner highlight along the nav icons row to match subtle elevation
draw.line([(0, nav_top+6), (w, nav_top+6)], fill=(250,250,251), width=1)

# Optional subtle left and right page margins shadow to frame content
frame_shadow_color = (245,245,247)
draw.rectangle([(0, header_bottom+10), (36, h-180)], fill=frame_shadow_color)
draw.rectangle([(w-36, header_bottom+10), (w, h-180)], fill=frame_shadow_color)

# End of structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/00_icon_DD_HEALTH.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1182), _c0)
except Exception:
    pass
layout["DD_HEALTH"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/01_icon_MAY_11_2024.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1974), _c1)
except Exception:
    pass
layout["MAY_11,_2024"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/02_icon_Yoga_sessiong.png
try:
    _c2 = get_crop(2, 1344, 191)
    canvas.paste(_c2, (48, 72), _c2)
except Exception:
    pass
layout["Yoga_sessiong"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/03_icon_Mnelennn.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1578), _c3)
except Exception:
    pass
layout["Mnelennn"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/04_icon_YOGA.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 390), _c4)
except Exception:
    pass
layout["YOGA"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/05_icon_YES_GIRLS_CREATE_AND_IAM_YOGI_STUDIOS.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 1974), _c5)
except Exception:
    pass
layout["YES_GIRLS_CREATE_AND_IAM_"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 52, 59)
    canvas.paste(_c6, (315, 4), _c6)
except Exception:
    pass
layout["icon_6"] = [315, 4, 367, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/07_icon_7.29.png
try:
    _c7 = get_crop(7, 51, 62)
    canvas.paste(_c7, (185, 3), _c7)
except Exception:
    pass
layout["7.29"] = [185, 3, 236, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 42, 56)
    canvas.paste(_c8, (254, 6), _c8)
except Exception:
    pass
layout["icon_8"] = [254, 6, 296, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/09_icon_7.29.png
try:
    _c9 = get_crop(9, 57, 62)
    canvas.paste(_c9, (114, 2), _c9)
except Exception:
    pass
layout["7.29"] = [114, 2, 171, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/10_icon_Yoga_Sessions_at_AZULIK.png
try:
    _c10 = get_crop(10, 1344, 396)
    canvas.paste(_c10, (48, 390), _c10)
except Exception:
    pass
layout["Yoga_Sessions_at_AZULIK"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/11_icon_Jersey.png
try:
    _c11 = get_crop(11, 1344, 396)
    canvas.paste(_c11, (48, 786), _c11)
except Exception:
    pass
layout["Jersey"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 40, 66)
    canvas.paste(_c12, (1159, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [1159, 1, 1199, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/13_icon_8_21_creator_followers.png
try:
    _c13 = get_crop(13, 1344, 396)
    canvas.paste(_c13, (48, 786), _c13)
except Exception:
    pass
layout["8_21_creator_followers"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/14_icon_Cancel.png
try:
    _c14 = get_crop(14, 76, 66)
    canvas.paste(_c14, (1217, 0), _c14)
except Exception:
    pass
layout["Cancel"] = [1217, 0, 1293, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/15_icon_Tickets.png
try:
    _c15 = get_crop(15, 288, 156)
    canvas.paste(_c15, (864, 2804), _c15)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/16_icon_7.29.png
try:
    _c16 = get_crop(16, 117, 103)
    canvas.paste(_c16, (57, 119), _c16)
except Exception:
    pass
layout["7.29"] = [57, 119, 174, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/17_icon_Cancel.png
try:
    _c17 = get_crop(17, 51, 65)
    canvas.paste(_c17, (1320, 0), _c17)
except Exception:
    pass
layout["Cancel"] = [1320, 0, 1371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/18_icon_Graf_Center_for_Integrative_Medicine.png
try:
    _c18 = get_crop(18, 1344, 396)
    canvas.paste(_c18, (48, 1182), _c18)
except Exception:
    pass
layout["Graf_Center_for_Integrati"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/19_icon_Cancel.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1099, 96), _c19)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/20_icon_Search_events.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (288, 2804), _c20)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/21_icon_Favorites.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (576, 2804), _c21)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/22_icon_More.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (1152, 2804), _c22)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/23_icon_Am_Yogi_Studios.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 1974), _c23)
except Exception:
    pass
layout["Am_Yogi_Studios"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/24_icon_Cancel.png
try:
    _c24 = get_crop(24, 149, 144)
    canvas.paste(_c24, (1243, 97), _c24)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/25_icon_Home.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (0, 2804), _c25)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/26_icon_8_00_AM_EST.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 390), _c26)
except Exception:
    pass
layout["8:00_AM_EST"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/27_icon_AZULIK_Tulum.png
try:
    _c27 = get_crop(27, 228, 53)
    canvas.paste(_c27, (390, 594), _c27)
except Exception:
    pass
layout["AZULIK_Tulum"] = [390, 594, 618, 647]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/28_icon_Cancel.png
try:
    _c28 = get_crop(28, 43, 63)
    canvas.paste(_c28, (1271, 2), _c28)
except Exception:
    pass
layout["Cancel"] = [1271, 2, 1314, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/29_icon_Yoga_sessiong.png
try:
    _c29 = get_crop(29, 43, 59)
    canvas.paste(_c29, (385, 4), _c29)
except Exception:
    pass
layout["Yoga_sessiong"] = [385, 4, 428, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/30_icon_Graf_Center_for_Integrative_Medicine.png
try:
    _c30 = get_crop(30, 1344, 396)
    canvas.paste(_c30, (48, 1578), _c30)
except Exception:
    pass
layout["Graf_Center_for_Integrati"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/31_text_7.29.png
try:
    _c31 = get_crop(31, 91, 45)
    canvas.paste(_c31, (20, 15), _c31)
except Exception:
    pass
layout["7.29"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/32_text_Events.png
try:
    _c32 = get_crop(32, 186, 56)
    canvas.paste(_c32, (46, 301), _c32)
except Exception:
    pass
layout["Events"] = [46, 301, 232, 357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/33_text_Sat.png
try:
    _c33 = get_crop(33, 77, 45)
    canvas.paste(_c33, (390, 2030), _c33)
except Exception:
    pass
layout["Sat,"] = [390, 2030, 467, 2075]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/34_text_11.png
try:
    _c34 = get_crop(34, 57, 36)
    canvas.paste(_c34, (542, 2034), _c34)
except Exception:
    pass
layout["11"] = [542, 2034, 599, 2070]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/35_text_12_00_PM_EDT.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 1974), _c35)
except Exception:
    pass
layout["12:00_PM_EDT"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_08_2024_4_23_19_27_45f56b06f31541079045047b6d542613-10/36_text_8_12_creator_followers.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 1974), _c36)
except Exception:
    pass
layout["8_12_creator_followers"] = [48, 1974, 1392, 2370]
