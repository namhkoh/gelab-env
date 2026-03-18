# page_id: page_eventbrite_182a9ea236924e9eb43ccbe82fd1506e_01
# screenshot: 2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3.png
# step_index: 1/13
# task: Open Eventbrite. Set time to tomorrow. Clear all search filters. Select the third one in New York. Record its location and time in Google Keep Notes. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided. Use them directly.

w, h = canvas.size

# Overall background (match the app's mostly white background with a very subtle warm tint)
draw.rectangle([(0, 0), (w, h)], fill="#fbfbfc")

# Status bar area (top ~72px) - light gray bar with subtle divider
status_h = 72
draw.rectangle([(0, 0), (w, status_h)], fill="#d0d0d0")
draw.line([(0, status_h), (w, status_h)], fill="#b8b8b8", width=1)

# Header / toolbar area (under status bar) - keep it clean and slightly elevated
header_top = status_h
header_bottom = 220
draw.rectangle([(0, header_top), (w, header_bottom)], fill="#ffffff")
# subtle bottom divider under header
draw.line([(48, header_bottom), (w-48, header_bottom)], fill="#efeff2", width=1)

# Rounded card backgrounds for each event row (do not draw icons/text inside them)
card_left = 48
card_right = w - 48
card_height = 396
card_radius = 16
card_outline = "#e9e9ee"
card_fill = "#ffffff"

row_tops = [490, 886, 1282, 1678, 2074, 2470]
for y in row_tops:
    top = y
    bottom = y + card_height
    # subtle shadow layer (very light, slightly shifted down)
    shadow_offset = 6
    shadow_bbox = [card_left, top + shadow_offset, card_right, bottom + shadow_offset]
    draw.rounded_rectangle(shadow_bbox, radius=card_radius, fill="#f5f5f7")
    # main card
    bbox = [card_left, top, card_right, bottom]
    draw.rounded_rectangle(bbox, radius=card_radius, fill=card_fill, outline=card_outline, width=1)
    # divider line at bottom of card for separation (helps list readibility)
    draw.line([(card_left + 12, bottom), (card_right - 12, bottom)], fill="#f0f0f2", width=1)

# Additional thin separators between stacked cards (in areas not covered by card outlines)
for y in [ (row_tops[i] + card_height) for i in range(len(row_tops)-1) ]:
    draw.line([(card_left, y + 1), (card_right, y + 1)], fill="#f7f7f9", width=1)

# Large content/banner area backgrounds (for items that look like full-bleed images further down)
# Example: a darker full-width band behind some rows to hint image/content areas (kept subtle)
band_left = 0
band_right = w
band_positions = [
    (2000, 2120),  # subtle band behind lower-middle content area (if any)
]
for top, bottom in band_positions:
    draw.rectangle([(band_left, top), (band_right, bottom)], fill="#fcfcfd")

# Bottom navigation bar background and top divider (leave icons/outlines to be pasted)
nav_height = 160
nav_top = h - nav_height
draw.rectangle([(0, nav_top), (w, h)], fill="#ffffff")
# subtle top border/shadow for the nav bar
draw.line([(0, nav_top), (w, nav_top)], fill="#e6e6e8", width=1)
draw.rectangle([(0, nav_top), (w, nav_top+6)], fill="#f8f8fa")

# Done: background, status bar, header area, card backgrounds, separators, bottom nav background.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/00_icon_iORk.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2074), _c0)
except Exception:
    pass
layout["iORk"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/01_icon_ZDRTTZY.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 490), _c1)
except Exception:
    pass
layout["ZDRTTZY"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/02_icon_95_HEEEYIMI_UESK_EEudooz.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 886), _c2)
except Exception:
    pass
layout["95_HEEEYIMI_UESK_EEudooz"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/03_icon_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/04_icon_DL_NO_COVER_ALL_NIGHT.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 490), _c4)
except Exception:
    pass
layout["DL_(NO_COVER_ALL_NIGHT)"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/05_icon_free_Grief_and_Loss_Healing_Circle.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 1282), _c5)
except Exception:
    pass
layout["(free)_Grief_and_Loss_Hea"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/06_icon_The_DL.png
try:
    _c6 = get_crop(6, 144, 123)
    canvas.paste(_c6, (1140, 1951), _c6)
except Exception:
    pass
layout["The_DL"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/07_icon_Best_Rooftop_Lounge_NYC.png
try:
    _c7 = get_crop(7, 1344, 396)
    canvas.paste(_c7, (48, 1678), _c7)
except Exception:
    pass
layout["Best_Rooftop_Lounge_NYC"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/08_icon_Favorite_button.png
try:
    _c8 = get_crop(8, 144, 139)
    canvas.paste(_c8, (1140, 1539), _c8)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/09_icon_Favorite_button.png
try:
    _c9 = get_crop(9, 144, 123)
    canvas.paste(_c9, (1140, 763), _c9)
except Exception:
    pass
layout["Favorite_button"] = [1140, 763, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/10_icon_The_DL.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1140, 2347), _c10)
except Exception:
    pass
layout["The_DL"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/11_icon_The_DL.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (288, 2804), _c11)
except Exception:
    pass
layout["The_DL"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/12_icon_The_DL.png
try:
    _c12 = get_crop(12, 144, 123)
    canvas.paste(_c12, (1284, 1951), _c12)
except Exception:
    pass
layout["The_DL"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 139)
    canvas.paste(_c13, (1284, 1539), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 123)
    canvas.paste(_c14, (1284, 1159), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1159, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/15_icon_The_DL.png
try:
    _c15 = get_crop(15, 144, 123)
    canvas.paste(_c15, (1284, 2347), _c15)
except Exception:
    pass
layout["The_DL"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/16_icon_dtLaIct.png
try:
    _c16 = get_crop(16, 1344, 396)
    canvas.paste(_c16, (48, 1678), _c16)
except Exception:
    pass
layout["dtLaIct"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/17_icon_Favorite_button.png
try:
    _c17 = get_crop(17, 144, 123)
    canvas.paste(_c17, (1140, 1159), _c17)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1159, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 56, 56)
    canvas.paste(_c18, (247, 5), _c18)
except Exception:
    pass
layout["icon_18"] = [247, 5, 303, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 52, 58)
    canvas.paste(_c19, (183, 3), _c19)
except Exception:
    pass
layout["icon_19"] = [183, 3, 235, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/20_icon_Best_Rooftop_Lounge_NYC.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 886), _c20)
except Exception:
    pass
layout["Best_Rooftop_Lounge_NYC"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/21_icon_Overflow_menu_button.png
try:
    _c21 = get_crop(21, 144, 123)
    canvas.paste(_c21, (1284, 763), _c21)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 763, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/22_icon_Ary.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (0, 2804), _c22)
except Exception:
    pass
layout["Ary"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 47, 52)
    canvas.paste(_c23, (1321, 7), _c23)
except Exception:
    pass
layout["icon_23"] = [1321, 7, 1368, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/24_icon_New_York.png
try:
    _c24 = get_crop(24, 405, 117)
    canvas.paste(_c24, (518, 2651), _c24)
except Exception:
    pass
layout["New_York"] = [518, 2651, 923, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/25_icon_9.30.png
try:
    _c25 = get_crop(25, 94, 101)
    canvas.paste(_c25, (46, 120), _c25)
except Exception:
    pass
layout["9.30"] = [46, 120, 140, 221]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 64, 58)
    canvas.paste(_c26, (1211, 5), _c26)
except Exception:
    pass
layout["icon_26"] = [1211, 5, 1275, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 62, 58)
    canvas.paste(_c27, (311, 5), _c27)
except Exception:
    pass
layout["icon_27"] = [311, 5, 373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 48, 56)
    canvas.paste(_c28, (383, 7), _c28)
except Exception:
    pass
layout["icon_28"] = [383, 7, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 42, 56)
    canvas.paste(_c29, (1272, 5), _c29)
except Exception:
    pass
layout["icon_29"] = [1272, 5, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/30_icon_Best_Rooftop_Lounge_NYC.png
try:
    _c30 = get_crop(30, 1344, 396)
    canvas.paste(_c30, (48, 2074), _c30)
except Exception:
    pass
layout["Best_Rooftop_Lounge_NYC"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/31_icon_TUmU_5i0.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (576, 2804), _c31)
except Exception:
    pass
layout["TUmU'5i0"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/32_icon_Fireworks_July_Ath_Rooftop_Party.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 2074), _c32)
except Exception:
    pass
layout["Fireworks_July_Ath_Roofto"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/33_icon_icon_33.png
try:
    _c33 = get_crop(33, 31, 48)
    canvas.paste(_c33, (913, 2687), _c33)
except Exception:
    pass
layout["icon_33"] = [913, 2687, 944, 2735]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/34_text_9.30.png
try:
    _c34 = get_crop(34, 94, 45)
    canvas.paste(_c34, (17, 15), _c34)
except Exception:
    pass
layout["9.30"] = [17, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/35_text_More_events_you_II_love.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 490), _c35)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/36_text_Sun_Jun_23.png
try:
    _c36 = get_crop(36, 205, 49)
    canvas.paste(_c36, (388, 2554), _c36)
except Exception:
    pass
layout["Sun,_Jun_23"] = [388, 2554, 593, 2603]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/37_text_3_00_PM_EDT.png
try:
    _c37 = get_crop(37, 1344, 346)
    canvas.paste(_c37, (48, 2470), _c37)
except Exception:
    pass
layout["3:00_PM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/38_text_The_DL_Rooftop.png
try:
    _c38 = get_crop(38, 144, 123)
    canvas.paste(_c38, (1140, 2347), _c38)
except Exception:
    pass
layout["The_DL_Rooftop"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/39_text_Ary.png
try:
    _c39 = get_crop(39, 1344, 346)
    canvas.paste(_c39, (48, 2470), _c39)
except Exception:
    pass
layout["Ary"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/40_text_The_DL.png
try:
    _c40 = get_crop(40, 115, 38)
    canvas.paste(_c40, (394, 2693), _c40)
except Exception:
    pass
layout["The_DL"] = [394, 2693, 509, 2731]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/41_text_TUmU_5i0.png
try:
    _c41 = get_crop(41, 405, 117)
    canvas.paste(_c41, (518, 2651), _c41)
except Exception:
    pass
layout["TUmU'5i0"] = [518, 2651, 923, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/42_clickable_Tickets.png
try:
    _c42 = get_crop(42, 288, 156)
    canvas.paste(_c42, (864, 2804), _c42)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_01_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-3/43_clickable_More.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (1152, 2804), _c43)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
