# page_id: page_eventbrite_b28077cf24f341ff9de826ac8bd7fb2b_13
# screenshot: 2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15.png
# step_index: 13/16
# task: Open Eventbrite. Explore 'Wellness' events in Washington. Filter to only show free events. Add the first non-promoted event to favorite and follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background with a very light gray to match the app's canvas
draw.rectangle([(0, 0), canvas.size], fill="#f5f6f7")

# Status bar (top ~96px) - subtle darker gray strip
status_bar_h = 96
draw.rectangle([(0, 0), (1440, status_bar_h)], fill="#e6e7e8")

# Header area below status bar (white)
header_top = status_bar_h
header_bottom = 220
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#ffffff")

# Thin divider under header
divider_y = header_bottom
draw.line([(40, divider_y), (1400, divider_y)], fill="#d6d7db", width=2)

# Filter / chips container area (light background to separate from header)
filters_top = divider_y + 8
filters_bottom = filters_top + 86
draw.rectangle([(0, filters_top), (1440, filters_bottom)], fill="#fbfcfd")
# subtle bottom divider for filters area
draw.line([(40, filters_bottom), (1400, filters_bottom)], fill="#e2e3e6", width=1)

# Function to draw subtle card shadow + rounded card
def draw_card(x0, y0, x1, y1, radius=28, card_fill="#ffffff", shadow_color="#e9eaec", shadow_offset=(0, 6)):
    # shadow
    sx0, sy0 = x0 + shadow_offset[0], y0 + shadow_offset[1]
    sx1, sy1 = x1 + shadow_offset[0], y1 + shadow_offset[1]
    draw.rounded_rectangle([(sx0, sy0), (sx1, sy1)], radius=radius, fill=shadow_color)
    # card
    draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=radius, fill=card_fill)

# First event card - large white rounded card (the image & text will be pasted on top)
card_margin_x = 48
card1_top = filters_bottom + 40
card1_bottom = 1450
draw_card(card_margin_x, card1_top, 1440 - card_margin_x, card1_bottom, radius=24)

# Inside first card: light image placeholder band (actual event image will be pasted over this)
img1_top = card1_top + 20
img1_bottom = img1_top + 320
draw.rectangle([(card_margin_x + 20, img1_top), (1440 - card_margin_x - 20, img1_bottom)], fill="#f2f3f5")

# Subtle separator under the image area inside the card
draw.line([(card_margin_x + 20, img1_bottom + 18), (1440 - card_margin_x - 20, img1_bottom + 18)], fill="#eef0f2", width=1)

# Second event card further down
card2_top = 1520
card2_bottom = 2500
draw_card(card_margin_x, card2_top, 1440 - card_margin_x, card2_bottom, radius=24)

# Inside second card: image placeholder
img2_top = card2_top + 20
img2_bottom = img2_top + 320
draw.rectangle([(card_margin_x + 20, img2_top), (1440 - card_margin_x - 20, img2_bottom)], fill="#f6f6f8")

# Divider lines between content sections (subtle)
sep1_y = card1_bottom + 18
draw.line([(40, sep1_y), (1400, sep1_y)], fill="#f0f1f3", width=1)
sep2_y = card2_bottom + 10
draw.line([(40, sep2_y), (1400, sep2_y)], fill="#f0f1f3", width=1)

# Bottom navigation bar background (approx same height as detected icons area)
nav_top = 2720
nav_bottom = 2960
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill="#ffffff")
# top border for nav
draw.line([(0, nav_top), (1440, nav_top)], fill="#e6e7e9", width=2)

# Small home bar hint (very thin, centered) to mimic modern nav handle
handle_w = 120
handle_h = 6
handle_x0 = (1440 - handle_w) // 2
handle_y0 = nav_top + 10
draw.rounded_rectangle([(handle_x0, handle_y0), (handle_x0 + handle_w, handle_y0 + handle_h)], radius=3, fill="#efeff1")

# Final subtle overall vignette at content edges (very light)
edge_shade = "#fafafa"
draw.rectangle([(0, nav_bottom - 6), (1440, nav_bottom)], fill=edge_shade)
draw.rectangle([(0, 0), (6, 2960)], fill=edge_shade)
draw.rectangle([(1440 - 6, 0), (1440, 2960)], fill=edge_shade)

# Note: Actual text, icons and images will be pasted on top at their detected coordinates.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/00_icon_Anytime.png
try:
    _c0 = get_crop(0, 1344, 191)
    canvas.paste(_c0, (48, 72), _c0)
except Exception:
    pass
layout["Anytime"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/01_icon_Music.png
try:
    _c1 = get_crop(1, 198, 112)
    canvas.paste(_c1, (843, 406), _c1)
except Exception:
    pass
layout["Music"] = [843, 406, 1041, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/02_icon_Business.png
try:
    _c2 = get_crop(2, 251, 113)
    canvas.paste(_c2, (1042, 405), _c2)
except Exception:
    pass
layout["Business"] = [1042, 405, 1293, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 493, 144)
    canvas.paste(_c3, (0, 259), _c3)
except Exception:
    pass
layout["1_Filter"] = [0, 259, 493, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/04_icon_Fo.png
try:
    _c4 = get_crop(4, 140, 110)
    canvas.paste(_c4, (1296, 406), _c4)
except Exception:
    pass
layout["Fo("] = [1296, 406, 1436, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/05_icon_26.png
try:
    _c5 = get_crop(5, 1344, 965)
    canvas.paste(_c5, (48, 525), _c5)
except Exception:
    pass
layout["26"] = [48, 525, 1392, 1490]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/06_icon_VA_22046.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 831), _c6)
except Exception:
    pass
layout["VA_22046"] = [1092, 831, 1236, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 831), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 831, 1380, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 52, 66)
    canvas.paste(_c8, (1152, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1152, 0, 1204, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/09_icon_Close_current_screen.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/10_icon_4.45.png
try:
    _c10 = get_crop(10, 115, 109)
    canvas.paste(_c10, (60, 115), _c10)
except Exception:
    pass
layout["4.45"] = [60, 115, 175, 224]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/11_icon_Wellness.png
try:
    _c11 = get_crop(11, 66, 64)
    canvas.paste(_c11, (308, 0), _c11)
except Exception:
    pass
layout["Wellness"] = [308, 0, 374, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/12_icon_4.45.png
try:
    _c12 = get_crop(12, 59, 64)
    canvas.paste(_c12, (181, 0), _c12)
except Exception:
    pass
layout["4.45"] = [181, 0, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/13_icon_Favorite_button.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1092, 2054), _c13)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2054, 1236, 2198]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 91, 63)
    canvas.paste(_c14, (1212, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1212, 0, 1303, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 51, 63)
    canvas.paste(_c15, (249, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [249, 0, 300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/16_icon_Wellness_in_Action_Core_Work.png
try:
    _c16 = get_crop(16, 1344, 1029)
    canvas.paste(_c16, (48, 1538), _c16)
except Exception:
    pass
layout["Wellness_in_Action:_Core_"] = [48, 1538, 1392, 2567]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/17_icon_4.45.png
try:
    _c17 = get_crop(17, 59, 65)
    canvas.paste(_c17, (115, 0), _c17)
except Exception:
    pass
layout["4.45"] = [115, 0, 174, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 55, 62)
    canvas.paste(_c18, (1318, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1318, 0, 1373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/19_icon_Wellness.png
try:
    _c19 = get_crop(19, 1344, 191)
    canvas.paste(_c19, (48, 72), _c19)
except Exception:
    pass
layout["Wellness"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/20_icon_Washington.png
try:
    _c20 = get_crop(20, 493, 144)
    canvas.paste(_c20, (0, 259), _c20)
except Exception:
    pass
layout["Washington"] = [0, 259, 493, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/21_icon_Tickets.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (864, 2804), _c21)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/22_icon_Overflow_menu_button.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1236, 2054), _c22)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2054, 1380, 2198]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/23_icon_welless.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (288, 2804), _c23)
except Exception:
    pass
layout["welless"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/24_icon_Wellness.png
try:
    _c24 = get_crop(24, 48, 61)
    canvas.paste(_c24, (384, 2), _c24)
except Exception:
    pass
layout["Wellness"] = [384, 2, 432, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/25_icon_108_Founders_Ave.png
try:
    _c25 = get_crop(25, 44, 59)
    canvas.paste(_c25, (284, 1386), _c25)
except Exception:
    pass
layout["108_Founders_Ave"] = [284, 1386, 328, 1445]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/26_icon_welless.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (576, 2804), _c26)
except Exception:
    pass
layout["welless"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/27_icon_Judea_Mu_ipr.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (864, 2804), _c27)
except Exception:
    pass
layout["Judea_Mu_ipr"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/28_icon_welless.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (576, 2804), _c28)
except Exception:
    pass
layout["welless"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 42, 62)
    canvas.paste(_c29, (1273, 0), _c29)
except Exception:
    pass
layout["icon_29"] = [1273, 0, 1315, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/30_icon_Promoted.png
try:
    _c30 = get_crop(30, 245, 64)
    canvas.paste(_c30, (83, 1383), _c30)
except Exception:
    pass
layout["Promoted"] = [83, 1383, 328, 1447]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/31_icon_More.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (1152, 2804), _c31)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/32_icon_Home.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (0, 2804), _c32)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/33_icon_Judea_Mu_ipr.png
try:
    _c33 = get_crop(33, 288, 156)
    canvas.paste(_c33, (1152, 2804), _c33)
except Exception:
    pass
layout["Judea_Mu_ipr"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/34_text_4.45.png
try:
    _c34 = get_crop(34, 92, 43)
    canvas.paste(_c34, (22, 15), _c34)
except Exception:
    pass
layout["4.45"] = [22, 15, 114, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/35_text_Free.png
try:
    _c35 = get_crop(35, 80, 39)
    canvas.paste(_c35, (117, 1030), _c35)
except Exception:
    pass
layout["Free"] = [117, 1030, 197, 1069]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/36_text_Grand_Opening_Party_4Ever_Young_Falls.png
try:
    _c36 = get_crop(36, 1344, 965)
    canvas.paste(_c36, (48, 525), _c36)
except Exception:
    pass
layout["Grand_Opening_Party_4Ever"] = [48, 525, 1392, 1490]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/37_text_Fri_Apr_26.png
try:
    _c37 = get_crop(37, 210, 52)
    canvas.paste(_c37, (90, 1258), _c37)
except Exception:
    pass
layout["Fri,_Apr_26"] = [90, 1258, 300, 1310]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/38_text_5.00_PM_EDT.png
try:
    _c38 = get_crop(38, 251, 45)
    canvas.paste(_c38, (320, 1259), _c38)
except Exception:
    pass
layout["5.00_PM_EDT"] = [320, 1259, 571, 1304]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_13_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-15/39_text_108_Founders_Ave.png
try:
    _c39 = get_crop(39, 341, 48)
    canvas.paste(_c39, (93, 1325), _c39)
except Exception:
    pass
layout["108_Founders_Ave"] = [93, 1325, 434, 1373]
