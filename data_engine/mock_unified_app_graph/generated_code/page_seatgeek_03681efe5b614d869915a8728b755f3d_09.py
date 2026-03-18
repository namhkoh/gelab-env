# page_id: page_seatgeek_03681efe5b614d869915a8728b755f3d_09
# screenshot: 2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12.png
# step_index: 9/10
# task: Open SeatGeek. Search "Metropolitan Opera". Find the next available show. Filter by "best seats". What section are they in for the lowest price tickets?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI elements for the described mobile UI.
# Uses provided canvas (1440x2960) and draw (ImageDraw) objects and fonts.

w, h = canvas.size

# Colors
bg_fill = (236, 239, 241)         # overall app background (light cool gray)
status_bar_fill = (34, 36, 38)    # status bar (dark)
toolbar_shadow = (220, 222, 224)  # subtle shadow under toolbar/search area
seatmap_bg = (215, 218, 221)      # seat map background
seatmap_outline = (195, 198, 200)
sheet_shadow = (210, 212, 214)
sheet_fill = (255, 255, 255)
card_fill = (250, 250, 250)
card_border = (222, 224, 226)
selected_border = (15, 15, 15)
handle_fill = (200, 203, 206)
divider = (224, 226, 228)

# Fill full background
draw.rectangle([(0, 0), (w, h)], fill=bg_fill)

# Status bar area (top ~72px)
status_h = 72
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_fill)

# subtle divider/shadow under the top area (below status bar)
draw.line([(0, status_h), (w, status_h)], fill=toolbar_shadow, width=1)

# Large seat-map/content area (center of screen)
# Keep this below the filter/pill area (pills are ~y=300), so start around y=360
seat_left = 120
seat_top = 360
seat_right = w - 120
seat_bottom = 1720
seat_radius = 18

# Light shadow for seatmap
shadow_offset = 8
draw.rounded_rectangle(
    [(seat_left + shadow_offset, seat_top + shadow_offset),
     (seat_right + shadow_offset, seat_bottom + shadow_offset)],
    radius=seat_radius + 2,
    fill=(225, 227, 229)
)

# Seatmap background rounded card
draw.rounded_rectangle(
    [(seat_left, seat_top), (seat_right, seat_bottom)],
    radius=seat_radius,
    fill=seatmap_bg,
    outline=seatmap_outline,
    width=2
)

# Add subtle inner highlight band near top of seatmap (background decoration only)
band_h = 42
draw.rectangle(
    [(seat_left + 12, seat_top + 12), (seat_right - 12, seat_top + 12 + band_h)],
    fill=(228, 231, 234)
)

# Horizontal divider line below filter pills area (visual separation)
divider_y = seat_top - 40
draw.line([(60, divider_y), (w - 60, divider_y)], fill=divider, width=1)

# Bottom sheet (rounded white panel)
sheet_margin = 60
sheet_top = 1840
sheet_radius = 40

# Sheet shadow (behind the sheet)
draw.rounded_rectangle(
    [(sheet_margin - 6, sheet_top + 10), (w - sheet_margin + 6, h - 10)],
    radius=sheet_radius + 4,
    fill=sheet_shadow
)

# White sheet
draw.rounded_rectangle(
    [(sheet_margin, sheet_top), (w - sheet_margin, h - 20)],
    radius=sheet_radius,
    fill=sheet_fill,
    outline=(240, 240, 240),
    width=1
)

# Handle (small rounded capsule) centered near top of sheet
handle_w = 120
handle_h = 8
handle_x0 = (w - handle_w) // 2
handle_y0 = sheet_top + 18
draw.rounded_rectangle(
    [(handle_x0, handle_y0), (handle_x0 + handle_w, handle_y0 + handle_h)],
    radius=handle_h // 2,
    fill=handle_fill
)

# "Sort by" sheet header divider (subtle)
header_div_y = sheet_top + 80
draw.line([(sheet_margin + 24, header_div_y), (w - sheet_margin - 24, header_div_y)], fill=(245,245,245), width=1)

# Cards inside bottom sheet: three stacked rounded rectangles (backgrounds only)
card_left = sheet_margin + 24
card_right = w - sheet_margin - 24
card_width = card_right - card_left
card_height_1 = 190
card_spacing = 30

# Card 1 (Deal Score)
card1_top = sheet_top + 100
card1_bottom = card1_top + card_height_1
draw.rounded_rectangle(
    [(card_left, card1_top), (card_right, card1_bottom)],
    radius=20,
    fill=card_fill,
    outline=card_border,
    width=2
)

# Card 2 (Price) - show selected state with darker outline
card2_top = card1_bottom + card_spacing
card2_bottom = card2_top + card_height_1
draw.rounded_rectangle(
    [(card_left, card2_top), (card_right, card2_bottom)],
    radius=20,
    fill=card_fill,
    outline=card_border,
    width=2
)
# darker outer stroke to indicate selection (subtle, outside the card by 2px)
draw.rounded_rectangle(
    [(card_left+2, card2_top+2), (card_right-2, card2_bottom-2)],
    radius=18,
    outline=selected_border,
    width=3
)

# Card 3 (Best Seats)
card3_top = card2_bottom + card_spacing
card3_bottom = card3_top + card_height_1
draw.rounded_rectangle(
    [(card_left, card3_top), (card_right, card3_bottom)],
    radius=20,
    fill=card_fill,
    outline=card_border,
    width=2
)

# Thin separators between cards (subtle)
sep_x0 = card_left + 12
sep_x1 = card_right - 12
draw.line([(sep_x0, card1_bottom + card_spacing//2), (sep_x1, card1_bottom + card_spacing//2)], fill=(245,245,245), width=1)
draw.line([(sep_x0, card2_bottom + card_spacing//2), (sep_x1, card2_bottom + card_spacing//2)], fill=(245,245,245), width=1)

# Small decorative lines on sheet (do not place over detected text areas)
# Add a thin top accent under the sheet header center
accent_y = sheet_top + 62
draw.line([(w//2 - 80, accent_y), (w//2 + 80, accent_y)], fill=(245,245,245), width=2)

# Final subtle vignette along the sides of canvas to match screenshot tint
side_vignette = (244, 245, 246)
vignette_w = 28
draw.rectangle([(0, 0), (vignette_w, h)], fill=side_vignette)
draw.rectangle([(w - vignette_w, 0), (w, h)], fill=side_vignette)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/00_icon_We_rate_each_ticket_based_on_location_pr.png
try:
    _c0 = get_crop(0, 1320, 267)
    canvas.paste(_c0, (60, 2318), _c0)
except Exception:
    pass
layout["We_rate_each_ticket_based"] = [60, 2318, 1380, 2585]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/01_icon_Include.png
try:
    _c1 = get_crop(1, 341, 118)
    canvas.paste(_c1, (537, 309), _c1)
except Exception:
    pass
layout["Include"] = [537, 309, 878, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/02_icon_Best_seats.png
try:
    _c2 = get_crop(2, 309, 118)
    canvas.paste(_c2, (910, 309), _c2)
except Exception:
    pass
layout["Best_seats"] = [910, 309, 1219, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/03_icon_Quantity.png
try:
    _c3 = get_crop(3, 279, 120)
    canvas.paste(_c3, (231, 307), _c3)
except Exception:
    pass
layout["Quantity"] = [231, 307, 510, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/04_icon_Tit.png
try:
    _c4 = get_crop(4, 169, 121)
    canvas.paste(_c4, (37, 308), _c4)
except Exception:
    pass
layout["Tit"] = [37, 308, 206, 429]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/05_icon_Low_pri.png
try:
    _c5 = get_crop(5, 193, 119)
    canvas.paste(_c5, (1247, 308), _c5)
except Exception:
    pass
layout["Low_pri"] = [1247, 308, 1440, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/06_icon_We_rate_each_ticket_based_on_location_pr.png
try:
    _c6 = get_crop(6, 1320, 329)
    canvas.paste(_c6, (60, 1941), _c6)
except Exception:
    pass
layout["We_rate_each_ticket_based"] = [60, 1941, 1380, 2270]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/07_icon_Include.png
try:
    _c7 = get_crop(7, 1353, 166)
    canvas.paste(_c7, (40, 118), _c7)
except Exception:
    pass
layout["Include"] = [40, 118, 1393, 284]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/08_icon_El_Nino.png
try:
    _c8 = get_crop(8, 64, 63)
    canvas.paste(_c8, (241, 2), _c8)
except Exception:
    pass
layout["El_Nino"] = [241, 2, 305, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/09_icon_7.58_my.png
try:
    _c9 = get_crop(9, 68, 64)
    canvas.paste(_c9, (110, 0), _c9)
except Exception:
    pass
layout["7.58_my"] = [110, 0, 178, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/10_icon_El_Nino.png
try:
    _c10 = get_crop(10, 60, 63)
    canvas.paste(_c10, (311, 2), _c10)
except Exception:
    pass
layout["El_Nino"] = [311, 2, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/11_icon_7.58_my.png
try:
    _c11 = get_crop(11, 54, 63)
    canvas.paste(_c11, (182, 1), _c11)
except Exception:
    pass
layout["7.58_my"] = [182, 1, 236, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 53, 56)
    canvas.paste(_c12, (1319, 4), _c12)
except Exception:
    pass
layout["icon_12"] = [1319, 4, 1372, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 52, 65)
    canvas.paste(_c13, (1152, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [1152, 1, 1204, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 104, 61)
    canvas.paste(_c14, (1213, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [1213, 2, 1317, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/15_icon_New_York.png
try:
    _c15 = get_crop(15, 47, 65)
    canvas.paste(_c15, (384, 1), _c15)
except Exception:
    pass
layout["New_York"] = [384, 1, 431, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/16_icon_Sort_by.png
try:
    _c16 = get_crop(16, 1320, 329)
    canvas.paste(_c16, (60, 1941), _c16)
except Exception:
    pass
layout["Sort_by"] = [60, 1941, 1380, 2270]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/17_icon_Low_pri.png
try:
    _c17 = get_crop(17, 96, 112)
    canvas.paste(_c17, (1258, 146), _c17)
except Exception:
    pass
layout["Low_pri"] = [1258, 146, 1354, 258]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/18_text_STAGE.png
try:
    _c18 = get_crop(18, 42, 16)
    canvas.paste(_c18, (470, 611), _c18)
except Exception:
    pass
layout["STAGE"] = [470, 611, 512, 627]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/19_text_ORCHESTRA_PIT.png
try:
    _c19 = get_crop(19, 136, 25)
    canvas.paste(_c19, (421, 684), _c19)
except Exception:
    pass
layout["ORCHESTRA_PIT"] = [421, 684, 557, 709]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/20_text_ORCH_L.png
try:
    _c20 = get_crop(20, 87, 27)
    canvas.paste(_c20, (354, 821), _c20)
except Exception:
    pass
layout["ORCH_L"] = [354, 821, 441, 848]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/21_text_49.png
try:
    _c21 = get_crop(21, 36, 27)
    canvas.paste(_c21, (844, 920), _c21)
except Exception:
    pass
layout["49"] = [844, 920, 880, 947]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/22_text_18.png
try:
    _c22 = get_crop(22, 34, 29)
    canvas.paste(_c22, (1057, 911), _c22)
except Exception:
    pass
layout["18"] = [1057, 911, 1091, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/23_text_27.png
try:
    _c23 = get_crop(23, 36, 29)
    canvas.paste(_c23, (923, 941), _c23)
except Exception:
    pass
layout["27"] = [923, 941, 959, 970]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/24_text_26.png
try:
    _c24 = get_crop(24, 34, 27)
    canvas.paste(_c24, (983, 939), _c24)
except Exception:
    pass
layout["26"] = [983, 939, 1017, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/25_text_LEVEL.png
try:
    _c25 = get_crop(25, 57, 25)
    canvas.paste(_c25, (402, 1017), _c25)
except Exception:
    pass
layout["LEVEL"] = [402, 1017, 459, 1042]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/26_text_ORCHESTRA.png
try:
    _c26 = get_crop(26, 106, 25)
    canvas.paste(_c26, (469, 1017), _c26)
except Exception:
    pass
layout["ORCHESTRA"] = [469, 1017, 575, 1042]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/27_text_LEVEL_2_PARTERRE.png
try:
    _c27 = get_crop(27, 159, 25)
    canvas.paste(_c27, (879, 1017), _c27)
except Exception:
    pass
layout["LEVEL_2_PARTERRE"] = [879, 1017, 1038, 1042]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/28_text_33.png
try:
    _c28 = get_crop(28, 31, 27)
    canvas.paste(_c28, (331, 1149), _c28)
except Exception:
    pass
layout["33"] = [331, 1149, 362, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/29_text_ROOM.png
try:
    _c29 = get_crop(29, 64, 27)
    canvas.paste(_c29, (950, 1367), _c29)
except Exception:
    pass
layout["ROOM"] = [950, 1367, 1014, 1394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/30_text_Best_Seats.png
try:
    _c30 = get_crop(30, 269, 55)
    canvas.paste(_c30, (118, 2703), _c30)
except Exception:
    pass
layout["Best_Seats"] = [118, 2703, 387, 2758]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/31_text_Get_close_to_the_action_with_the_best_se.png
try:
    _c31 = get_crop(31, 1320, 267)
    canvas.paste(_c31, (60, 2633), _c31)
except Exception:
    pass
layout["Get_close_to_the_action_w"] = [60, 2633, 1380, 2900]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/32_text_STANDING.png
try:
    _c32 = get_crop(32, 102, 46)
    canvas.paste(_c32, (539, 933), _c32)
except Exception:
    pass
layout["STANDING"] = [539, 933, 641, 979]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/33_text_STANDING.png
try:
    _c33 = get_crop(33, 104, 46)
    canvas.paste(_c33, (338, 934), _c33)
except Exception:
    pass
layout["STANDING"] = [338, 934, 442, 980]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/34_text_STANDING.png
try:
    _c34 = get_crop(34, 101, 66)
    canvas.paste(_c34, (790, 1300), _c34)
except Exception:
    pass
layout["STANDING"] = [790, 1300, 891, 1366]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/35_text_STANDING.png
try:
    _c35 = get_crop(35, 103, 60)
    canvas.paste(_c35, (322, 1317), _c35)
except Exception:
    pass
layout["STANDING"] = [322, 1317, 425, 1377]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_09_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-12/36_text_STANDING.png
try:
    _c36 = get_crop(36, 104, 56)
    canvas.paste(_c36, (554, 1319), _c36)
except Exception:
    pass
layout["~STANDING"] = [554, 1319, 658, 1375]
