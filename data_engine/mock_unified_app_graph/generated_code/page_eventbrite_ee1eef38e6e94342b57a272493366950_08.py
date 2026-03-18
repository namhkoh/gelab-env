# page_id: page_eventbrite_ee1eef38e6e94342b57a272493366950_08
# screenshot: 2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10.png
# step_index: 8/10
# task: Open Eventbrite. Open "Fashion" category. Apply filter for free events. From the list, select the first non-promoted event and add it to your favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill
draw.rectangle([(0, 0), canvas.size], fill="#F6F7FA")

# Status bar (top ~80px)
status_h = 80
draw.rectangle([(0, 0), (1440, status_h)], fill="#AFAFB4")

# Header / Search area background
header_top = status_h
header_bottom = 264  # covers the large search area region
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")

# Subtle rounded search-bar backdrop (behind pasted search content)
search_x0, search_x1 = 48, 1392
search_y0, search_y1 = 96, 220
draw.rounded_rectangle([(search_x0, search_y0), (search_x1, search_y1)], radius=28, fill="#FBFDFF", outline=None)

# Divider under header / search area
draw.line([(48, header_bottom), (1392, header_bottom)], fill="#E6E7EA", width=2)

# Light divider above filters area
filters_div_y = 336
draw.line([(48, filters_div_y), (1392, filters_div_y)], fill="#F0F0F3", width=1)

# Card shadow/backdrop for first event card (subtle grey block behind)
card1_x0, card1_x1 = 36, 1404
card1_y0, card1_y1 = 600, 1800
# shadow
draw.rounded_rectangle([(card1_x0+6, card1_y0+10), (card1_x1+6, card1_y1+10)], radius=28, fill="#EFEFF1")
# card background
draw.rounded_rectangle([(card1_x0, card1_y0), (card1_x1, card1_y1)], radius=28, fill="#FFFFFF", outline="#E8EAEE")

# Thin separator below first card
sep_y = card1_y1 + 16
draw.line([(48, sep_y), (1392, sep_y)], fill="#ECEEF1", width=1)

# Card shadow/backdrop for second event card
card2_x0, card2_x1 = 36, 1404
card2_y0, card2_y1 = 1800, 2840
draw.rounded_rectangle([(card2_x0+6, card2_y0+10), (card2_x1+6, card2_y1+10)], radius=28, fill="#EFEFF1")
draw.rounded_rectangle([(card2_x0, card2_y0), (card2_x1, card2_y1)], radius=28, fill="#FFFFFF", outline="#E8EAEE")

# Dark content background for image areas (so pasted images have a darker panel behind)
# First image area (matches the large event image region background)
img1_x0, img1_y0 = 48, 684
img1_x1, img1_y1 = img1_x0 + 1344, img1_y0 + 1108
draw.rectangle([(img1_x0, img1_y0), (img1_x1, img1_y1)], fill="#F8F9FB")

# Second image area (darker banner behind second event image)
img2_x0, img2_y0 = 48, 1840
img2_x1, img2_y1 = img2_x0 + 1344, img2_y0 + 976
draw.rectangle([(img2_x0, img2_y0), (img2_x1, img2_y1)], fill="#2F2F34")

# Small subtle separators around image/card edges
draw.line([(48, img1_y1 + 16), (1392, img1_y1 + 16)], fill="#F1F2F4", width=1)
draw.line([(48, img2_y1 + 16), (1392, img2_y1 + 16)], fill="#F1F2F4", width=1)

# Bottom navigation bar background and divider
nav_top = 2880
draw.line([(0, nav_top), (1440, nav_top)], fill="#E6E7EA", width=1)
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#FFFFFF")

# Subtle rounded top edge highlight for nav
draw.rectangle([(0, nav_top), (1440, nav_top+6)], fill="#FAFBFC")

# Final thin global bottom border
draw.line([(0, 2959), (1440, 2959)], fill="#E9EAED", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/00_icon_Anytime.png
try:
    _c0 = get_crop(0, 1344, 111)
    canvas.paste(_c0, (48, 525), _c0)
except Exception:
    pass
layout["Anytime"] = [48, 525, 1392, 636]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/01_icon_Fashion.png
try:
    _c1 = get_crop(1, 232, 113)
    canvas.paste(_c1, (862, 405), _c1)
except Exception:
    pass
layout["Fashion"] = [862, 405, 1094, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/02_icon_2_Filters.png
try:
    _c2 = get_crop(2, 377, 144)
    canvas.paste(_c2, (0, 259), _c2)
except Exception:
    pass
layout["2_Filters"] = [0, 259, 377, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/03_icon_Buaston.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 1200), _c3)
except Exception:
    pass
layout["Buaston"] = [1092, 1200, 1236, 1344]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/04_icon_Buaston.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1236, 1200), _c4)
except Exception:
    pass
layout["Buaston"] = [1236, 1200, 1380, 1344]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2356), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2356, 1236, 2500]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2356), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2356, 1380, 2500]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/07_icon_Recognise_Respond_Refer_-.png
try:
    _c7 = get_crop(7, 1344, 1108)
    canvas.paste(_c7, (48, 684), _c7)
except Exception:
    pass
layout["Recognise,_Respond_&_Refe"] = [48, 684, 1392, 1792]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/08_icon_5.28.png
try:
    _c8 = get_crop(8, 122, 113)
    canvas.paste(_c8, (56, 114), _c8)
except Exception:
    pass
layout["5.28"] = [56, 114, 178, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/09_icon_5.28.png
try:
    _c9 = get_crop(9, 60, 64)
    canvas.paste(_c9, (181, 0), _c9)
except Exception:
    pass
layout["5.28"] = [181, 0, 241, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/10_icon_Search_forae.png
try:
    _c10 = get_crop(10, 69, 63)
    canvas.paste(_c10, (307, 0), _c10)
except Exception:
    pass
layout["Search_forae"] = [307, 0, 376, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 54, 64)
    canvas.paste(_c11, (246, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [246, 0, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/12_icon_5.28.png
try:
    _c12 = get_crop(12, 59, 65)
    canvas.paste(_c12, (115, 0), _c12)
except Exception:
    pass
layout["5.28"] = [115, 0, 174, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 69, 60)
    canvas.paste(_c13, (1206, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1206, 0, 1275, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 64, 59)
    canvas.paste(_c14, (1317, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1317, 0, 1381, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/15_icon_Online.png
try:
    _c15 = get_crop(15, 377, 144)
    canvas.paste(_c15, (0, 259), _c15)
except Exception:
    pass
layout["Online"] = [0, 259, 377, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/16_icon_Promoted.png
try:
    _c16 = get_crop(16, 1344, 111)
    canvas.paste(_c16, (48, 525), _c16)
except Exception:
    pass
layout["Promoted"] = [48, 525, 1392, 636]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/17_icon_Search_forae.png
try:
    _c17 = get_crop(17, 1344, 191)
    canvas.paste(_c17, (48, 72), _c17)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/18_icon_Recognise_Respond_Refer_-.png
try:
    _c18 = get_crop(18, 1344, 1108)
    canvas.paste(_c18, (48, 684), _c18)
except Exception:
    pass
layout["Recognise,_Respond_&_Refe"] = [48, 684, 1392, 1792]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/19_icon_Earnyour.png
try:
    _c19 = get_crop(19, 1344, 976)
    canvas.paste(_c19, (48, 1840), _c19)
except Exception:
    pass
layout["Earnyour"] = [48, 1840, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/20_icon_F_REE_HG.png
try:
    _c20 = get_crop(20, 1344, 1108)
    canvas.paste(_c20, (48, 684), _c20)
except Exception:
    pass
layout["F_REE_HG"] = [48, 684, 1392, 1792]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/21_icon_on.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (288, 2804), _c21)
except Exception:
    pass
layout["on"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/22_icon_Search_forae.png
try:
    _c22 = get_crop(22, 50, 62)
    canvas.paste(_c22, (384, 1), _c22)
except Exception:
    pass
layout["Search_forae"] = [384, 1, 434, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 39, 61)
    canvas.paste(_c23, (1275, 0), _c23)
except Exception:
    pass
layout["icon_23"] = [1275, 0, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/24_icon_DAa_CDT.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["DAa_CDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/25_icon_PRINCIPLES_OF.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (864, 2804), _c25)
except Exception:
    pass
layout["PRINCIPLES_OF"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/26_text_5.28.png
try:
    _c26 = get_crop(26, 91, 45)
    canvas.paste(_c26, (20, 15), _c26)
except Exception:
    pass
layout["5.28"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/27_text_Online.png
try:
    _c27 = get_crop(27, 128, 48)
    canvas.paste(_c27, (93, 1695), _c27)
except Exception:
    pass
layout["Online"] = [93, 1695, 221, 1743]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/28_text_Free.png
try:
    _c28 = get_crop(28, 80, 37)
    canvas.paste(_c28, (117, 2556), _c28)
except Exception:
    pass
layout["Free"] = [117, 2556, 197, 2593]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/29_text_REDKEN_CANADA.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (288, 2804), _c29)
except Exception:
    pass
layout["REDKEN_CANADA"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/30_text_PRINCIPLES_OF.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (576, 2804), _c30)
except Exception:
    pass
layout["PRINCIPLES_OF"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/31_text_HAIRCOLOR.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (0, 2804), _c31)
except Exception:
    pass
layout["HAIRCOLOR"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/32_text_on.png
try:
    _c32 = get_crop(32, 50, 25)
    canvas.paste(_c32, (400, 2784), _c32)
except Exception:
    pass
layout["on"] = [400, 2784, 450, 2809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/33_text_DAa_CDT.png
try:
    _c33 = get_crop(33, 141, 18)
    canvas.paste(_c33, (461, 2790), _c33)
except Exception:
    pass
layout["DAa_CDT"] = [461, 2790, 602, 2808]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_08_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-10/34_clickable_More.png
try:
    _c34 = get_crop(34, 288, 156)
    canvas.paste(_c34, (1152, 2804), _c34)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
