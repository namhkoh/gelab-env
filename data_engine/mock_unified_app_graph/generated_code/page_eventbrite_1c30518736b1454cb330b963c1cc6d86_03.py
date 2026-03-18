# page_id: page_eventbrite_1c30518736b1454cb330b963c1cc6d86_03
# screenshot: 2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5.png
# step_index: 3/9
# task: Open Eventbrite. Search for "Open Mic Nights". Filter the results to only include free events. Select the first non-promoted event in the list - what"s the location of that event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI drawing for Eventbrite-style search results
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
bg_color = (255, 255, 255)             # main page background (white)
status_bar_color = (200, 200, 200)     # light gray status bar
header_underline_color = (34, 90, 255) # vivid blue for search underline
card_fill = (255, 255, 255)            # card background white
card_border = (230, 230, 235)          # very light border for cards
separator = (240, 240, 245)            # separators between rows
nav_bar_color = (255, 255, 255)        # bottom nav background (white)
shadow_color = (220, 220, 225)         # subtle shadow / divider color

W, H = canvas.size

# Fill whole background (canvas may already be white but ensure consistency)
draw.rectangle([0, 0, W, H], fill=bg_color)

# Status bar (top area) - leave icons/text to be pasted later
status_bar_h = 56
draw.rectangle([0, 0, W, status_bar_h], fill=status_bar_color)

# Header area (search input region)
header_top = status_bar_h
header_bottom = 220
draw.rectangle([0, header_top, W, header_bottom], fill=bg_color)

# Blue underline for the search field (across content width with margins similar to UI)
underline_left = 48
underline_right = W - 48
underline_y = header_bottom - 6
underline_thickness = 6
draw.rectangle([underline_left, underline_y, underline_right, underline_y + underline_thickness], fill=header_underline_color)

# Subtle 1px divider under the underline (slight shadow)
draw.line([(underline_left, underline_y + underline_thickness + 2), (underline_right, underline_y + underline_thickness + 2)], fill=shadow_color, width=1)

# Section title area divider (above list "Events")
section_div_y = 300
draw.line([(48, section_div_y), (W - 48, section_div_y)], fill=separator, width=1)

# Draw rounded card backgrounds for each detected event block (do not draw any text or icons)
# Using detected top positions and sizes from the UI crop metadata.
event_cards = [
    (48, 72, 48 + 1344, 72 + 191),    # featured/first item (shorter)
    (48, 390, 48 + 1344, 390 + 396),
    (48, 786, 48 + 1344, 786 + 396),
    (48, 1182, 48 + 1344, 1182 + 396),
    (48, 1578, 48 + 1344, 1578 + 396),
    (48, 1974, 48 + 1344, 1974 + 396),
    (48, 2370, 48 + 1344, 2370 + 396),
]

card_radius = 14
for (x1, y1, x2, y2) in event_cards:
    # card fill
    draw.rounded_rectangle([x1, y1, x2, y2], radius=card_radius, fill=card_fill, outline=card_border, width=1)
    # thin separator line at bottom of each card for extra clarity
    draw.line([(x1 + 8, y2), (x2 - 8, y2)], fill=separator, width=1)

# Light vertical alignment guide (visual spacing) - subtle left gutter guideline (not content)
gutter_x = 48
draw.line([(gutter_x - 0.5, header_bottom), (gutter_x - 0.5, H - 200)], fill=(245,245,250), width=1)

# Bottom navigation bar background and top divider
nav_top = 2804
nav_bottom = H
draw.rectangle([0, nav_top, W, nav_bottom], fill=nav_bar_color)
# top border of nav
draw.line([(0, nav_top), (W, nav_top)], fill=shadow_color, width=1)

# Small shadow under last list item area to separate from nav
draw.rectangle([48, nav_top - 12, W - 48, nav_top - 11], fill=shadow_color)

# Additional separators between list rows (in case content heights vary)
# We place separators at the bottoms of the event_cards (already drawn), and add a few extras for spacing
extra_seps = [320, 716, 1112, 1508, 1904, 2296]
for y in extra_seps:
    draw.line([(48, y), (W - 48, y)], fill=(245,245,248), width=1)

# subtle left thumbnail background boxes (slot for thumbnails) - drawn as rounded squares behind where thumbnails will be pasted
# Thumbnails in the UI are at left within each card; draw a light rounded box to suggest image container (but not the image itself)
thumb_w = 176
thumb_h_small = 176
thumb_h_large = 120
thumb_positions = [
    (48 + 8, 72 + 8, 48 + 8 + thumb_h_large, 72 + 8 + thumb_h_large),   # featured small square
    (48 + 8, 390 + 8, 48 + 8 + thumb_w, 390 + 8 + thumb_w),
    (48 + 8, 786 + 8, 48 + 8 + thumb_w, 786 + 8 + thumb_w),
    (48 + 8, 1182 + 8, 48 + 8 + thumb_w, 1182 + 8 + thumb_w),
    (48 + 8, 1578 + 8, 48 + 8 + thumb_w, 1578 + 8 + thumb_w),
    (48 + 8, 1974 + 8, 48 + 8 + thumb_w, 1974 + 8 + thumb_w),
    (48 + 8, 2370 + 8, 48 + 8 + thumb_w, 2370 + 8 + thumb_w),
]
for (tx1, ty1, tx2, ty2) in thumb_positions:
    # very light neutral box to give structure behind pasted thumbnails (keeps from duplicating any image content)
    draw.rounded_rectangle([tx1, ty1, tx2, ty2], radius=8, fill=(250,250,250), outline=(240,240,245), width=1)

# final subtle horizontal rule near top to separate status/header from content
draw.line([(0, header_bottom + 6), (W, header_bottom + 6)], fill=separator, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/00_icon_DeSOcIA.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 390), _c0)
except Exception:
    pass
layout["DeSOcIA"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/01_icon_niGht_UT.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 786), _c1)
except Exception:
    pass
layout["niGht_@UT"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/02_icon_Open_Mic_Night.png
try:
    _c2 = get_crop(2, 1344, 191)
    canvas.paste(_c2, (48, 72), _c2)
except Exception:
    pass
layout["Open_Mic_Night"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/03_icon_Staud_U_Gowet.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1578), _c3)
except Exception:
    pass
layout["[Staud_U?_Gowet"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/04_icon_GET_Ol.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 1974), _c4)
except Exception:
    pass
layout["GET_Ol"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 52, 60)
    canvas.paste(_c5, (315, 3), _c5)
except Exception:
    pass
layout["icon_5"] = [315, 3, 367, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/06_icon_4.53.png
try:
    _c6 = get_crop(6, 51, 62)
    canvas.paste(_c6, (185, 2), _c6)
except Exception:
    pass
layout["4.53"] = [185, 2, 236, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 42, 57)
    canvas.paste(_c7, (254, 5), _c7)
except Exception:
    pass
layout["icon_7"] = [254, 5, 296, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/08_icon_4.53.png
try:
    _c8 = get_crop(8, 58, 62)
    canvas.paste(_c8, (114, 2), _c8)
except Exception:
    pass
layout["4.53"] = [114, 2, 172, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/09_icon_fadvanced_Standup_Cod.png
try:
    _c9 = get_crop(9, 1344, 396)
    canvas.paste(_c9, (48, 2370), _c9)
except Exception:
    pass
layout["fadvanced_Standup_Cod"] = [48, 2370, 1392, 2766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/10_icon_OPEN_IMI.png
try:
    _c10 = get_crop(10, 1344, 396)
    canvas.paste(_c10, (48, 1182), _c10)
except Exception:
    pass
layout["OPEN_IMI"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/11_icon_Open_Mic_Night.png
try:
    _c11 = get_crop(11, 1344, 396)
    canvas.paste(_c11, (48, 390), _c11)
except Exception:
    pass
layout["Open_Mic_Night"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/12_icon_4.53.png
try:
    _c12 = get_crop(12, 112, 101)
    canvas.paste(_c12, (60, 120), _c12)
except Exception:
    pass
layout["4.53"] = [60, 120, 172, 221]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/13_icon_6_00_PM_PDT.png
try:
    _c13 = get_crop(13, 1344, 396)
    canvas.paste(_c13, (48, 1974), _c13)
except Exception:
    pass
layout["6:00_PM_PDT"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/14_icon_Center_South.png
try:
    _c14 = get_crop(14, 1344, 396)
    canvas.paste(_c14, (48, 1182), _c14)
except Exception:
    pass
layout["Center_South"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/15_icon_Cancel.png
try:
    _c15 = get_crop(15, 79, 63)
    canvas.paste(_c15, (1216, 0), _c15)
except Exception:
    pass
layout["Cancel"] = [1216, 0, 1295, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/16_icon_Cancel.png
try:
    _c16 = get_crop(16, 49, 60)
    canvas.paste(_c16, (1321, 2), _c16)
except Exception:
    pass
layout["Cancel"] = [1321, 2, 1370, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/17_icon_8810_creator_followers.png
try:
    _c17 = get_crop(17, 288, 156)
    canvas.paste(_c17, (288, 2804), _c17)
except Exception:
    pass
layout["8810_creator_followers"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/18_icon_Higher_Mic_Open_Mic_Night.png
try:
    _c18 = get_crop(18, 1344, 396)
    canvas.paste(_c18, (48, 1578), _c18)
except Exception:
    pass
layout["Higher_Mic:_Open_Mic_Nigh"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/19_icon_8_4230_creator_followers.png
try:
    _c19 = get_crop(19, 1344, 396)
    canvas.paste(_c19, (48, 390), _c19)
except Exception:
    pass
layout["8_4230_creator_followers"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/20_icon_COLLAB_Dance_Studio_Creative_Space.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 786), _c20)
except Exception:
    pass
layout["COLLAB_Dance_Studio_&_Cre"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/21_icon_6_00_PM_PDT.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 1974), _c21)
except Exception:
    pass
layout["6:00_PM_PDT"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/22_icon_Cancel.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1099, 96), _c22)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/23_icon_Fire_Open_Mic_Nights.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 1182), _c23)
except Exception:
    pass
layout["Fire_Open_Mic_Nights!"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/24_icon_Tickets.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (864, 2804), _c24)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/25_icon_sletted_co_bighcrmic.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (0, 2804), _c25)
except Exception:
    pass
layout["~sletted_co_bighcrmic"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/26_icon_Open_Mic_Night_Out_Whistle_While_You_Hea.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 786), _c26)
except Exception:
    pass
layout["Open_Mic_Night_Out_(Whist"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/27_icon_6_00_PM_PDT.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 1578), _c27)
except Exception:
    pass
layout["6:00_PM_PDT"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/28_icon_Cancel.png
try:
    _c28 = get_crop(28, 42, 62)
    canvas.paste(_c28, (1272, 2), _c28)
except Exception:
    pass
layout["Cancel"] = [1272, 2, 1314, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 44, 59)
    canvas.paste(_c29, (385, 4), _c29)
except Exception:
    pass
layout["icon_29"] = [385, 4, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/30_icon_Cancel.png
try:
    _c30 = get_crop(30, 149, 144)
    canvas.paste(_c30, (1243, 97), _c30)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/31_icon_8810_creator_followers.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (576, 2804), _c31)
except Exception:
    pass
layout["8810_creator_followers"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/32_icon_Center_South.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 1182), _c32)
except Exception:
    pass
layout["Center_South"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/33_icon_The_Green_Room_on_Ventura.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 1578), _c33)
except Exception:
    pass
layout["The_Green_Room_on_Ventura"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/34_icon_More.png
try:
    _c34 = get_crop(34, 288, 156)
    canvas.paste(_c34, (1152, 2804), _c34)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/35_icon_The_EVEN_Higher_Mic_Open_Mic_Night.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 2370), _c35)
except Exception:
    pass
layout["The_EVEN_Higher_Mic:_Open"] = [48, 2370, 1392, 2766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/36_icon_Expression_Mondays.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 1974), _c36)
except Exception:
    pass
layout["Expression_Mondays"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/37_text_4.53.png
try:
    _c37 = get_crop(37, 89, 43)
    canvas.paste(_c37, (22, 17), _c37)
except Exception:
    pass
layout["4.53"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/38_text_Events.png
try:
    _c38 = get_crop(38, 191, 65)
    canvas.paste(_c38, (44, 299), _c38)
except Exception:
    pass
layout["Events"] = [44, 299, 235, 364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/39_text_The_EVEN_Higher_Mic_Open_Mic_Night.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 2370), _c39)
except Exception:
    pass
layout["The_EVEN_Higher_Mic:_Open"] = [48, 2370, 1392, 2766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/40_text_The_Green_Room_on_Ventura.png
try:
    _c40 = get_crop(40, 1344, 396)
    canvas.paste(_c40, (48, 2370), _c40)
except Exception:
    pass
layout["The_Green_Room_on_Ventura"] = [48, 2370, 1392, 2766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_03_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-5/41_text_8810_creator_followers.png
try:
    _c41 = get_crop(41, 1344, 396)
    canvas.paste(_c41, (48, 2370), _c41)
except Exception:
    pass
layout["8810_creator_followers"] = [48, 2370, 1392, 2766]
