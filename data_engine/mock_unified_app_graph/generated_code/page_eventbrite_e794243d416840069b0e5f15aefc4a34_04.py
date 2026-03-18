# page_id: page_eventbrite_e794243d416840069b0e5f15aefc4a34_04
# screenshot: 2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6.png
# step_index: 4/7
# task: Open Eventbrite. Open "Business Seminar". Select the first event. Note the contact details of the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill (dominant page color)
draw.rectangle([(0, 0), (1440, 2960)], fill="#fbfbfc")

# Status bar area (top)
draw.rectangle([(0, 0), (1440, 96)], fill="#cfcfcf")

# Header / toolbar area (under status bar)
draw.rectangle([(0, 96), (1440, 264)], fill="#ffffff")
# Thin divider under header
draw.line([(48, 264), (1392, 264)], fill="#dedede", width=2)
draw.line([(48, 268), (1392, 268)], fill="#f5f5f6", width=1)

# Event listing background separator bar (subtle)
draw.line([(48, 340), (1392, 340)], fill="#f0f0f2", width=1)

# First event card container (rounded)
c_left, c_right = 48, 1392
card1_top, card1_bottom = 360, 860
draw.rounded_rectangle([c_left, card1_top, c_right, card1_bottom],
                       radius=24, fill="#ffffff", outline="#e9e9eb", width=1)

# Placeholder banner area inside first card (light neutral background for image area)
draw.rounded_rectangle([c_left+12, card1_top+12, c_right-12, card1_top+176],
                       radius=14, fill="#f3f5f8")

# Subtle divider inside first card (between image area and details)
draw.line([(c_left+20, card1_top+192), (c_right-20, card1_top+192)], fill="#efeff1", width=1)

# Second event card with colored banner background (dark/teal area behind posted image)
card2_top, card2_bottom = 920, 1460
# Outer card background
draw.rounded_rectangle([c_left, card2_top, c_right, card2_bottom],
                       radius=24, fill="#073a36")
# Inner slightly different teal to create a bordered banner look
draw.rounded_rectangle([c_left+12, card2_top+12, c_right-12, card2_bottom-12],
                       radius=20, fill="#0d4b46")

# Divider lines between event cards and surrounding areas
draw.line([(48, 880), (1392, 880)], fill="#efeff1", width=1)
draw.line([(48, 930), (1392, 930)], fill="#efeff1", width=1)

# Small white content strip below second card to receive pasted text content (keeps contrast)
draw.rectangle([c_left, card2_bottom+16, c_right, card2_bottom+120], fill="#ffffff")

# Bottom navigation area (bar background and top divider)
nav_top, nav_bottom = 2760, 2960
draw.line([(0, nav_top), (1440, nav_top)], fill="#e6e6e6", width=2)
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill="#ffffff")

# Subtle page side shadows to add depth to cards (thin stripes)
draw.rectangle([(c_left-8, card1_top+8), (c_left-4, card1_bottom+8)], fill="#f5f6f7")
draw.rectangle([(c_right+4, card1_top+8), (c_right+8, card1_bottom+8)], fill="#f5f6f7")
draw.rectangle([(c_left-8, card2_top+8), (c_left-4, card2_bottom+8)], fill="#0b352f")
draw.rectangle([(c_right+4, card2_top+8), (c_right+8, card2_bottom+8)], fill="#0b352f")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 400, 103)
    canvas.paste(_c1, (425, 410), _c1)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/02_icon_Business.png
try:
    _c2 = get_crop(2, 241, 103)
    canvas.paste(_c2, (1036, 410), _c2)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/04_icon_STEPHANIE_BOHR.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2336), _c4)
except Exception:
    pass
layout["STEPHANIE_BOHR"] = [1092, 2336, 1236, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/05_icon_Foo.png
try:
    _c5 = get_crop(5, 139, 110)
    canvas.paste(_c5, (1284, 406), _c5)
except Exception:
    pass
layout["Foo"] = [1284, 406, 1423, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/06_icon_STEPHANIE_BOHR.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2336), _c6)
except Exception:
    pass
layout["STEPHANIE_BOHR"] = [1236, 2336, 1380, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/07_icon_Close_current_screen.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1248, 96), _c7)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/08_icon_5.20.png
try:
    _c8 = get_crop(8, 126, 119)
    canvas.paste(_c8, (54, 111), _c8)
except Exception:
    pass
layout["5.20"] = [54, 111, 180, 230]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/09_icon_5.20.png
try:
    _c9 = get_crop(9, 63, 64)
    canvas.paste(_c9, (179, 0), _c9)
except Exception:
    pass
layout["5.20"] = [179, 0, 242, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 68, 62)
    canvas.paste(_c10, (307, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [307, 1, 375, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 104, 61)
    canvas.paste(_c11, (1207, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1207, 0, 1311, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 52, 62)
    canvas.paste(_c12, (249, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [249, 1, 301, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/13_icon_Jo.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1092, 1192), _c13)
except Exception:
    pass
layout["Jo"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/14_icon_5.20.png
try:
    _c14 = get_crop(14, 59, 65)
    canvas.paste(_c14, (116, 0), _c14)
except Exception:
    pass
layout["5.20"] = [116, 0, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 65, 60)
    canvas.paste(_c15, (1317, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1317, 0, 1382, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/16_icon_Quinton_Sick.png
try:
    _c16 = get_crop(16, 1344, 1096)
    canvas.paste(_c16, (48, 676), _c16)
except Exception:
    pass
layout["Quinton_Sick"] = [48, 676, 1392, 1772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/17_icon_Quinton_Sick.png
try:
    _c17 = get_crop(17, 1344, 1096)
    canvas.paste(_c17, (48, 676), _c17)
except Exception:
    pass
layout["Quinton_Sick"] = [48, 676, 1392, 1772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/18_icon_Online.png
try:
    _c18 = get_crop(18, 377, 144)
    canvas.paste(_c18, (0, 259), _c18)
except Exception:
    pass
layout["Online"] = [0, 259, 377, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/19_icon_First-Time_Home_Seller_Online_Seminar.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (864, 2804), _c19)
except Exception:
    pass
layout["First-Time_Home_Seller_On"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/20_icon_Business_Seminar.png
try:
    _c20 = get_crop(20, 1344, 191)
    canvas.paste(_c20, (48, 72), _c20)
except Exception:
    pass
layout["Business_Seminar"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 51, 61)
    canvas.paste(_c21, (383, 2), _c21)
except Exception:
    pass
layout["icon_21"] = [383, 2, 434, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 101, 108)
    canvas.paste(_c22, (1287, 1876), _c22)
except Exception:
    pass
layout["icon_22"] = [1287, 1876, 1388, 1984]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/23_icon_8.30_PM_EDT.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (288, 2804), _c23)
except Exception:
    pass
layout["8.30_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/24_icon_MORTGAGE.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (1152, 2804), _c24)
except Exception:
    pass
layout["MORTGAGE"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/25_icon_First-Time_Home_Seller_Online_Seminar.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["First-Time_Home_Seller_On"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/26_icon_Jo.png
try:
    _c26 = get_crop(26, 144, 144)
    canvas.paste(_c26, (1092, 1192), _c26)
except Exception:
    pass
layout["Jo"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/27_icon_Thu_Mav_9.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (0, 2804), _c27)
except Exception:
    pass
layout["Thu,_Mav_9"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/28_icon_Promoted.png
try:
    _c28 = get_crop(28, 42, 61)
    canvas.paste(_c28, (284, 1668), _c28)
except Exception:
    pass
layout["Promoted"] = [284, 1668, 326, 1729]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/29_icon_5.20.png
try:
    _c29 = get_crop(29, 164, 65)
    canvas.paste(_c29, (3, 0), _c29)
except Exception:
    pass
layout["5.20"] = [3, 0, 167, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/30_text_10_000_events.png
try:
    _c30 = get_crop(30, 359, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/31_text_11.png
try:
    _c31 = get_crop(31, 53, 41)
    canvas.paste(_c31, (269, 1546), _c31)
except Exception:
    pass
layout["11"] = [269, 1546, 322, 1587]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/32_text_Online.png
try:
    _c32 = get_crop(32, 131, 48)
    canvas.paste(_c32, (90, 1607), _c32)
except Exception:
    pass
layout["Online"] = [90, 1607, 221, 1655]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/33_text_UNLOCK_YOUR_HOME_S_SELLING_POTENTIAL.png
try:
    _c33 = get_crop(33, 1344, 996)
    canvas.paste(_c33, (48, 1820), _c33)
except Exception:
    pass
layout["UNLOCK_YOUR_HOME'S_SELLIN"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/34_text_FiRST-TIME_HOME_SELLER_ONLINE_SEMINAR.png
try:
    _c34 = get_crop(34, 1344, 996)
    canvas.paste(_c34, (48, 1820), _c34)
except Exception:
    pass
layout["FiRST-TIME_HOME_SELLER_ON"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/35_text_Join_US_On_Zoomi.png
try:
    _c35 = get_crop(35, 374, 38)
    canvas.paste(_c35, (142, 2006), _c35)
except Exception:
    pass
layout["Join_US_On_Zoomi"] = [142, 2006, 516, 2044]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/36_text_ASHLEY_RUH.png
try:
    _c36 = get_crop(36, 246, 50)
    canvas.paste(_c36, (451, 2118), _c36)
except Exception:
    pass
layout["ASHLEY_RUH"] = [451, 2118, 697, 2168]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/37_text_REAL_Estate_Agent.png
try:
    _c37 = get_crop(37, 1344, 996)
    canvas.paste(_c37, (48, 1820), _c37)
except Exception:
    pass
layout["REAL_Estate_Agent"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/38_text_Free.png
try:
    _c38 = get_crop(38, 83, 42)
    canvas.paste(_c38, (115, 2534), _c38)
except Exception:
    pass
layout["Free"] = [115, 2534, 198, 2576]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/39_text_Unlock_Your_Home_s_Selling_Potential.png
try:
    _c39 = get_crop(39, 1344, 996)
    canvas.paste(_c39, (48, 1820), _c39)
except Exception:
    pass
layout["Unlock_Your_Home's_Sellin"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/40_text_First-Time_Home_Seller_Online_Seminar.png
try:
    _c40 = get_crop(40, 1344, 996)
    canvas.paste(_c40, (48, 1820), _c40)
except Exception:
    pass
layout["First-Time_Home_Seller_On"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/41_text_Thu_Mav_9.png
try:
    _c41 = get_crop(41, 288, 156)
    canvas.paste(_c41, (0, 2804), _c41)
except Exception:
    pass
layout["Thu,_Mav_9"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_04_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-6/42_clickable_Overflow_menu_button.png
try:
    _c42 = get_crop(42, 144, 144)
    canvas.paste(_c42, (1236, 1192), _c42)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]
