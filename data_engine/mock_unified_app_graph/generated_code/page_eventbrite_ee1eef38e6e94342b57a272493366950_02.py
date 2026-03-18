# page_id: page_eventbrite_ee1eef38e6e94342b57a272493366950_02
# screenshot: 2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4.png
# step_index: 2/10
# task: Open Eventbrite. Open "Fashion" category. Apply filter for free events. From the list, select the first non-promoted event and add it to your favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for Eventbrite-like mobile page
# (Uses provided canvas (1440x2960) and draw ImageDraw; font_sm/md/lg/xl available)

# Overall background
bg_color = (249, 250, 252)  # very light off-white
draw.rectangle([(0, 0), canvas.size], fill=bg_color)

# Status bar (top system bar)
status_h = 56
status_color = (183, 183, 183)  # medium gray bar
draw.rectangle([(0, 0), (canvas.width, status_h)], fill=status_color)

# Thin subtle divider under status bar
draw.line([(0, status_h), (canvas.width, status_h)], fill=(210, 210, 210), width=1)

# Search/header area background (rounded white card behind search controls)
search_x, search_y = 48, 72
search_w, search_h = 1344, 160  # match detected search region width
search_radius = 30
search_rect = [search_x, search_y, search_x + search_w, search_y + search_h]
# subtle shadow (offset)
shadow_offset = 6
draw.rounded_rectangle(
    [search_rect[0], search_rect[1] + shadow_offset, search_rect[2], search_rect[3] + shadow_offset],
    radius=search_radius, fill=(235, 236, 238)
)
# main search background
draw.rounded_rectangle(search_rect, radius=search_radius, fill=(255, 255, 255))

# Divider line below search area
divider_y = search_y + search_h + 18
draw.line([(48, divider_y), (canvas.width - 48, divider_y)], fill=(226, 226, 230), width=2)

# Filters/category chip area separator (chips themselves are pasted later; draw only subtle background band)
chips_band_top = divider_y + 8
chips_band_bottom = chips_band_top + 70
# very light translucent band to group the filter chips area
draw.rectangle([(48, chips_band_top), (canvas.width - 48, chips_band_bottom)], fill=(250, 251, 253))
# bottom separator under chips
draw.line([(48, chips_band_bottom + 10), (canvas.width - 48, chips_band_bottom + 10)], fill=(230, 230, 233), width=1)

# Primary content card container (large rounded area behind event cards)
# Use detected large card region at pos (48,676) size (1344x1175)
cards_x, cards_y = 48, 676
cards_w, cards_h = 1344, 1175
cards_radius = 28
# soft shadow for the card block
draw.rounded_rectangle(
    [cards_x, cards_y + 8, cards_x + cards_w, cards_y + cards_h + 8],
    radius=cards_radius, fill=(240, 241, 243)
)
# white card background
draw.rounded_rectangle([cards_x, cards_y, cards_x + cards_w, cards_y + cards_h], radius=cards_radius, fill=(255, 255, 255))

# First event image placeholder (rounded rectangle) - top portion of the card block
img_margin = 24
first_img_x0 = cards_x + img_margin
first_img_y0 = cards_y + img_margin
first_img_x1 = cards_x + cards_w - img_margin
first_img_y1 = first_img_y0 + 420  # approximate image height
# image card shadow
draw.rounded_rectangle(
    [first_img_x0, first_img_y0 + 6, first_img_x1, first_img_y1 + 6],
    radius=20, fill=(220, 220, 222)
)
# image background (subtle warm photo placeholder)
draw.rounded_rectangle([first_img_x0, first_img_y0, first_img_x1, first_img_y1], radius=20, fill=(233, 229, 224))

# First event's action button background placeholders (circular background where icons will be pasted)
# Draw only the subtle circular background shapes (no icons)
icon_bg_color = (255, 255, 255)
heart_circle_center = (1092 + 72, 1192)  # note: icons themselves will be pasted later; keep background circles subtle
# Instead of using exact icon positions (these will be pasted), draw faint circular badges behind image area where icons usually float
badge_radius = 44
# right-side badge group (two stacked badge circles)
badge_x = first_img_x1 - 88
badge_y_top = first_img_y1 - 62
draw.ellipse([(badge_x - badge_radius, badge_y_top - badge_radius), (badge_x + badge_radius, badge_y_top + badge_radius)], fill=icon_bg_color)
draw.ellipse([(badge_x + 70 - badge_radius, badge_y_top - badge_radius), (badge_x + 70 + badge_radius, badge_y_top + badge_radius)], fill=icon_bg_color)

# Subtle drop shadow under first image to separate from text area
draw.line([(first_img_x0, first_img_y1 + 18), (first_img_x1, first_img_y1 + 18)], fill=(240, 240, 242), width=6)

# Second event image placeholder (rounded rectangle) - lower portion within the same card container
second_img_y0 = first_img_y1 + 200
second_img_y1 = second_img_y0 + 420
draw.rounded_rectangle(
    [first_img_x0, second_img_y0 + 6, first_img_x1, second_img_y1 + 6],
    radius=20, fill=(200, 205, 216)
)
# darker image background (to match the blue promotional image)
draw.rounded_rectangle([first_img_x0, second_img_y0, first_img_x1, second_img_y1], radius=20, fill=(44, 63, 92))

# small "Free" tag background placeholders near image text (rounded pill behind where label will be pasted)
pill_w, pill_h = 72, 36
pill_x = first_img_x0
pill1_y = first_img_y1 + 26
draw.rounded_rectangle([pill_x, pill1_y, pill_x + pill_w, pill1_y + pill_h], radius=8, fill=(225, 237, 233))

pill2_y = second_img_y1 + 26
draw.rounded_rectangle([pill_x, pill2_y, pill_x + pill_w, pill2_y + pill_h], radius=8, fill=(225, 237, 233))

# Thin separators between event cards/sections (subtle)
sep_y1 = first_img_y1 + 140
sep_y2 = second_img_y1 + 140
draw.line([(48 + 8, sep_y1), (48 + cards_w - 8, sep_y1)], fill=(243, 243, 246), width=2)
draw.line([(48 + 8, sep_y2), (48 + cards_w - 8, sep_y2)], fill=(243, 243, 246), width=2)

# Bottom navigation bar background
nav_h = 96
nav_top = canvas.height - nav_h
draw.rectangle([(0, nav_top), (canvas.width, canvas.height)], fill=(255, 255, 255))
# top border line for nav
draw.line([(24, nav_top), (canvas.width - 24, nav_top)], fill=(220, 220, 223), width=2)

# small safe-area indicator (very subtle) above nav for content spacing
draw.line([(48, nav_top - 12), (canvas.width - 48, nav_top - 12)], fill=(245, 245, 247), width=1)

# Final subtle vignette around content edges to anchor center content
edge_shade = 6
draw.rectangle([(0, status_h), (24, canvas.height)], fill=(248, 249, 250))
draw.rectangle([(canvas.width - 24, status_h), (canvas.width, canvas.height)], fill=(248, 249, 250))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 1192), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2415), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 1192), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/07_icon_Foo.png
try:
    _c7 = get_crop(7, 149, 110)
    canvas.paste(_c7, (1282, 406), _c7)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1431, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2415), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/09_icon_5.27.png
try:
    _c9 = get_crop(9, 122, 112)
    canvas.paste(_c9, (57, 116), _c9)
except Exception:
    pass
layout["5.27"] = [57, 116, 179, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/10_icon_Search_forae.png
try:
    _c10 = get_crop(10, 71, 64)
    canvas.paste(_c10, (306, 0), _c10)
except Exception:
    pass
layout["Search_forae"] = [306, 0, 377, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/11_icon_5.27.png
try:
    _c11 = get_crop(11, 62, 65)
    canvas.paste(_c11, (113, 0), _c11)
except Exception:
    pass
layout["5.27"] = [113, 0, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/12_icon_5.27.png
try:
    _c12 = get_crop(12, 61, 64)
    canvas.paste(_c12, (181, 0), _c12)
except Exception:
    pass
layout["5.27"] = [181, 0, 242, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 54, 64)
    canvas.paste(_c13, (246, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [246, 0, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/14_icon_Tech_to_Franchise_Trailblazer_Mapping_Yo.png
try:
    _c14 = get_crop(14, 1344, 1175)
    canvas.paste(_c14, (48, 676), _c14)
except Exception:
    pass
layout["Tech_to_Franchise_Trailbl"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/15_icon_Search_forae.png
try:
    _c15 = get_crop(15, 1344, 191)
    canvas.paste(_c15, (48, 72), _c15)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 90, 60)
    canvas.paste(_c16, (1207, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1207, 0, 1297, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 64, 59)
    canvas.paste(_c17, (1315, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1315, 0, 1379, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/18_icon_Online.png
try:
    _c18 = get_crop(18, 377, 144)
    canvas.paste(_c18, (0, 259), _c18)
except Exception:
    pass
layout["Online"] = [0, 259, 377, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/19_icon_Tech_to_Franchise_Trailblazer_Mapping_Yo.png
try:
    _c19 = get_crop(19, 1344, 1175)
    canvas.paste(_c19, (48, 676), _c19)
except Exception:
    pass
layout["Tech_to_Franchise_Trailbl"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/20_icon_Scale._or_Streamline_vour_business.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (288, 2804), _c20)
except Exception:
    pass
layout["Scale._or_Streamline_vour"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/21_icon_Search_forae.png
try:
    _c21 = get_crop(21, 52, 62)
    canvas.paste(_c21, (383, 2), _c21)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 435, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/22_icon_Promoted.png
try:
    _c22 = get_crop(22, 244, 65)
    canvas.paste(_c22, (84, 1743), _c22)
except Exception:
    pass
layout["Promoted"] = [84, 1743, 328, 1808]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/23_icon_2022.png
try:
    _c23 = get_crop(23, 1344, 917)
    canvas.paste(_c23, (48, 1899), _c23)
except Exception:
    pass
layout["2022"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/24_icon_SSCG_Lunch_Learn_Looking_to_Launch.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (864, 2804), _c24)
except Exception:
    pass
layout["SSCG_Lunch_&_Learn:_Looki"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/25_icon_Free.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (0, 2804), _c25)
except Exception:
    pass
layout["Free"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/26_icon_Scale._or_Streamline_vour_business.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (576, 2804), _c26)
except Exception:
    pass
layout["Scale._or_Streamline_vour"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/27_icon_Free.png
try:
    _c27 = get_crop(27, 125, 77)
    canvas.paste(_c27, (91, 2592), _c27)
except Exception:
    pass
layout["Free"] = [91, 2592, 216, 2669]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/28_icon_5.27.png
try:
    _c28 = get_crop(28, 146, 63)
    canvas.paste(_c28, (7, 0), _c28)
except Exception:
    pass
layout["5.27"] = [7, 0, 153, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 39, 61)
    canvas.paste(_c29, (1275, 0), _c29)
except Exception:
    pass
layout["icon_29"] = [1275, 0, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/30_icon_SSCG_Lunch_Learn_Looking_to_Launch.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (1152, 2804), _c30)
except Exception:
    pass
layout["SSCG_Lunch_&_Learn:_Looki"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/31_icon_More.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (1152, 2804), _c31)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/32_text_10_000_events.png
try:
    _c32 = get_crop(32, 359, 103)
    canvas.paste(_c32, (54, 410), _c32)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/33_text_Online.png
try:
    _c33 = get_crop(33, 129, 45)
    canvas.paste(_c33, (91, 1687), _c33)
except Exception:
    pass
layout["Online"] = [91, 1687, 220, 1732]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/34_text_Join.png
try:
    _c34 = get_crop(34, 87, 29)
    canvas.paste(_c34, (518, 1961), _c34)
except Exception:
    pass
layout["Join"] = [518, 1961, 605, 1990]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/35_text_Ws.png
try:
    _c35 = get_crop(35, 50, 27)
    canvas.paste(_c35, (615, 1963), _c35)
except Exception:
    pass
layout["Ws"] = [615, 1963, 665, 1990]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/36_text_oR.png
try:
    _c36 = get_crop(36, 73, 27)
    canvas.paste(_c36, (673, 1963), _c36)
except Exception:
    pass
layout["oR"] = [673, 1963, 746, 1990]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_02_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-4/37_text_OUR_SMALL_BUSINESS.png
try:
    _c37 = get_crop(37, 1344, 917)
    canvas.paste(_c37, (48, 1899), _c37)
except Exception:
    pass
layout["OUR_SMALL_BUSINESS"] = [48, 1899, 1392, 2816]
