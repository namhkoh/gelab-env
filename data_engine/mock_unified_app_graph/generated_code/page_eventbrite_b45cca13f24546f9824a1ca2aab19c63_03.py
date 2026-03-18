# page_id: page_eventbrite_b45cca13f24546f9824a1ca2aab19c63_03
# screenshot: 2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5.png
# step_index: 3/11
# task: Open Eventbrite. Search for "Art". Filter for events in New York. Select first recommended event. Save it to wishlist. What is the duration of the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the provided canvas
# Available variables: canvas (PIL Image 1440x2960 RGB), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Overall background (slightly off-white to match screenshot)
draw.rectangle((0, 0, W, H), fill=(250, 250, 250))

# Status bar at top (~72px)
status_h = 72
draw.rectangle((0, 0, W, status_h), fill=(158, 158, 158))

# Header / toolbar area (below status bar)
header_top = status_h
header_bottom = 264  # approximate bottom of header area
draw.rectangle((0, header_top, W, header_bottom), fill=(255, 255, 255))

# Thin blue underline under the header (search underline)
underline_left = 48
underline_right = W - 48
underline_y = header_bottom + 0  # just below header
draw.rectangle((underline_left, underline_y, underline_right, underline_y + 4), fill=(43, 76, 255))

# A light divider below the popular/search area (to separate sections)
draw.line((48, 320, W - 48, 320), fill=(230, 230, 230), width=1)

# Section: Events list background spacing
# Draw rounded card backgrounds (with subtle shadow) for each detected event card area.
card_x = 48
card_w = 1344
card_h = 396
card_radius = 12
card_ys = [1117, 1513, 1909, 2305]  # top y coordinates from detected elements

for y in card_ys:
    # shadow (offset downwards)
    shadow_offset = 8
    draw.rounded_rectangle(
        (card_x + 2, y + shadow_offset, card_x + card_w + 2, y + card_h + shadow_offset),
        radius=card_radius,
        fill=(235, 235, 235),
        outline=None
    )
    # main card background
    draw.rounded_rectangle(
        (card_x, y, card_x + card_w, y + card_h),
        radius=card_radius,
        fill=(255, 255, 255),
        outline=(220, 220, 220)
    )
    # subtle separator line under each card to visually separate (light)
    draw.line((card_x + 12, y + card_h, card_x + card_w - 12, y + card_h), fill=(245, 245, 245), width=1)

# A faint horizontal rule above the events list to anchor the section
draw.line((48, 1088, W - 48, 1088), fill=(235, 235, 235), width=1)

# Content area accent: small light gray blocks to hint image slots on the left of each card
# (We draw them as structure/background only; actual thumbnails will be pasted on top.)
thumb_margin = 24
thumb_size = 150
for y in card_ys:
    tx = card_x + thumb_margin
    ty = y + 36
    # light rounded rectangle representing image container background (will be covered by pasted thumbnail)
    draw.rounded_rectangle(
        (tx, ty, tx + thumb_size, ty + thumb_size),
        radius=8,
        fill=(245, 245, 245),
        outline=(235, 235, 235)
    )

# Bottom navigation bar background & top divider
nav_top = 2804
draw.rectangle((0, nav_top, W, H), fill=(255, 255, 255))
# top divider line for nav
draw.line((0, nav_top, W, nav_top), fill=(220, 220, 220), width=2)
# subtle shadow just above nav to separate from content
draw.rectangle((0, nav_top - 6, W, nav_top), fill=(250, 250, 250))

# Final subtle full-width separator near the bottom of content (above nav)
draw.line((48, nav_top - 100, W - 48, nav_top - 100), fill=(240, 240, 240), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/00_icon_Online.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2305), _c0)
except Exception:
    pass
layout["Online"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/01_icon_Online.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1117), _c1)
except Exception:
    pass
layout["Online"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/02_icon_8_1340_creator_followers.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1117), _c2)
except Exception:
    pass
layout["8_1340_creator_followers"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 54, 59)
    canvas.paste(_c3, (314, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [314, 3, 368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/04_icon_7.04.png
try:
    _c4 = get_crop(4, 55, 62)
    canvas.paste(_c4, (116, 2), _c4)
except Exception:
    pass
layout["7.04"] = [116, 2, 171, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/05_icon_7.04.png
try:
    _c5 = get_crop(5, 53, 60)
    canvas.paste(_c5, (183, 2), _c5)
except Exception:
    pass
layout["7.04"] = [183, 2, 236, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 41, 54)
    canvas.paste(_c6, (254, 6), _c6)
except Exception:
    pass
layout["icon_6"] = [254, 6, 295, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/07_icon_The_ART_in_Articulation.png
try:
    _c7 = get_crop(7, 1344, 396)
    canvas.paste(_c7, (48, 1909), _c7)
except Exception:
    pass
layout["The_ART_in_Articulation"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/08_icon_Molday_UaV_6.png
try:
    _c8 = get_crop(8, 1344, 396)
    canvas.paste(_c8, (48, 1513), _c8)
except Exception:
    pass
layout["Molday,_UaV_6"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/09_icon_7.04.png
try:
    _c9 = get_crop(9, 115, 105)
    canvas.paste(_c9, (60, 119), _c9)
except Exception:
    pass
layout["7.04"] = [60, 119, 175, 224]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/10_icon_176_creator_followers.png
try:
    _c10 = get_crop(10, 1344, 396)
    canvas.paste(_c10, (48, 1513), _c10)
except Exception:
    pass
layout["176_creator_followers"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/11_icon_Tue_Apr_23_._7_00_PM_EDT.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (288, 2804), _c11)
except Exception:
    pass
layout["Tue,_Apr_23_._7:00_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/12_icon_Cancel.png
try:
    _c12 = get_crop(12, 46, 60)
    canvas.paste(_c12, (1323, 2), _c12)
except Exception:
    pass
layout["Cancel"] = [1323, 2, 1369, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/13_icon_Tue_Apr_23_._7_00_PM_EDT.png
try:
    _c13 = get_crop(13, 288, 156)
    canvas.paste(_c13, (576, 2804), _c13)
except Exception:
    pass
layout["Tue,_Apr_23_._7:00_PM_EDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/14_icon_Tickets.png
try:
    _c14 = get_crop(14, 288, 156)
    canvas.paste(_c14, (864, 2804), _c14)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/15_icon_Cancel.png
try:
    _c15 = get_crop(15, 83, 62)
    canvas.paste(_c15, (1216, 0), _c15)
except Exception:
    pass
layout["Cancel"] = [1216, 0, 1299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/16_icon_Cancel.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1099, 96), _c16)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/17_icon_Cancel.png
try:
    _c17 = get_crop(17, 42, 61)
    canvas.paste(_c17, (1272, 2), _c17)
except Exception:
    pass
layout["Cancel"] = [1272, 2, 1314, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/18_icon_Cancel.png
try:
    _c18 = get_crop(18, 149, 144)
    canvas.paste(_c18, (1243, 97), _c18)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/19_icon_8_18571_creator_followers.png
try:
    _c19 = get_crop(19, 1344, 396)
    canvas.paste(_c19, (48, 2305), _c19)
except Exception:
    pass
layout["8_18571_creator_followers"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/20_icon_art.png
try:
    _c20 = get_crop(20, 1344, 120)
    canvas.paste(_c20, (48, 378), _c20)
except Exception:
    pass
layout["art"] = [48, 378, 1392, 498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/21_icon_Art_Artists_The_Art_of_Modigliani.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 1117), _c21)
except Exception:
    pass
layout["Art_&_Artists:_The_Art_of"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/22_icon_art.png
try:
    _c22 = get_crop(22, 1344, 120)
    canvas.paste(_c22, (48, 738), _c22)
except Exception:
    pass
layout["art"] = [48, 738, 1392, 858]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/23_icon_1I_O0AM_EDT.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 1909), _c23)
except Exception:
    pass
layout["1I:O0AM_EDT"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/24_icon_Art_Writing_for_Art_Professionals.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 1513), _c24)
except Exception:
    pass
layout["Art_Writing_for_Art_Profe"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/25_icon_More.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (1152, 2804), _c25)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/26_icon_Home.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/27_icon_1I_O0AM_EDT.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 1909), _c27)
except Exception:
    pass
layout["1I:O0AM_EDT"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/28_icon_Online.png
try:
    _c28 = get_crop(28, 111, 53)
    canvas.paste(_c28, (391, 2541), _c28)
except Exception:
    pass
layout["Online"] = [391, 2541, 502, 2594]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/29_icon_7.04.png
try:
    _c29 = get_crop(29, 90, 60)
    canvas.paste(_c29, (17, 3), _c29)
except Exception:
    pass
layout["7.04"] = [17, 3, 107, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/30_icon_Art_Freel_Artful_Doodles_with_Rusty_Hard.png
try:
    _c30 = get_crop(30, 1344, 396)
    canvas.paste(_c30, (48, 2305), _c30)
except Exception:
    pass
layout["Art_Freel_Artful_Doodles_"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/31_icon_Online.png
try:
    _c31 = get_crop(31, 112, 52)
    canvas.paste(_c31, (390, 1719), _c31)
except Exception:
    pass
layout["Online"] = [390, 1719, 502, 1771]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/32_icon_Online.png
try:
    _c32 = get_crop(32, 112, 51)
    canvas.paste(_c32, (391, 2115), _c32)
except Exception:
    pass
layout["Online"] = [391, 2115, 503, 2166]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/33_icon_LECTURE_Wopkshop_With_FlutiST.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 1909), _c33)
except Exception:
    pass
layout["LECTURE_Wopkshop_With_Flu"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/34_icon_Art_Writing_for_Art_Professionals.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 1513), _c34)
except Exception:
    pass
layout["Art_Writing_for_Art_Profe"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/35_icon_art_history.png
try:
    _c35 = get_crop(35, 1344, 144)
    canvas.paste(_c35, (48, 858), _c35)
except Exception:
    pass
layout["art_history"] = [48, 858, 1392, 1002]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/36_icon_arts.png
try:
    _c36 = get_crop(36, 1344, 120)
    canvas.paste(_c36, (48, 618), _c36)
except Exception:
    pass
layout["arts"] = [48, 618, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/37_icon_art.png
try:
    _c37 = get_crop(37, 92, 91)
    canvas.paste(_c37, (33, 531), _c37)
except Exception:
    pass
layout["art"] = [33, 531, 125, 622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/38_text_Art.png
try:
    _c38 = get_crop(38, 126, 73)
    canvas.paste(_c38, (200, 135), _c38)
except Exception:
    pass
layout["Art"] = [200, 135, 326, 208]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/39_text_Popular.png
try:
    _c39 = get_crop(39, 221, 78)
    canvas.paste(_c39, (44, 298), _c39)
except Exception:
    pass
layout["Popular"] = [44, 298, 265, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/40_text_artificial_intelligence.png
try:
    _c40 = get_crop(40, 1344, 120)
    canvas.paste(_c40, (48, 498), _c40)
except Exception:
    pass
layout["artificial_intelligence"] = [48, 498, 1392, 618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/41_text_Events.png
try:
    _c41 = get_crop(41, 191, 61)
    canvas.paste(_c41, (45, 1026), _c41)
except Exception:
    pass
layout["Events"] = [45, 1026, 236, 1087]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/42_text_Tue_Apr_23_._7_00_PM_EDT.png
try:
    _c42 = get_crop(42, 288, 156)
    canvas.paste(_c42, (576, 2804), _c42)
except Exception:
    pass
layout["Tue,_Apr_23_._7:00_PM_EDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_03_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-5/43_clickable_Art.png
try:
    _c43 = get_crop(43, 1344, 191)
    canvas.paste(_c43, (48, 72), _c43)
except Exception:
    pass
layout["Art"] = [48, 72, 1392, 263]
