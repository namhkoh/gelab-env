# page_id: page_seatgeek_68e3462c14734440a7ace3fed432a10d_11
# screenshot: 2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14.png
# step_index: 11/13
# task: Open SeatGeek and change the current location to Los Angeles. Then find the first concert show and track its performer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint overall background to match the app's light neutral canvas
draw.rectangle([(0, 0), (1440, 2960)], fill="#F7F7F7")

# Status bar (top strip) - subtle light gray band
status_h = 80
draw.rectangle([(0, 0), (1440, status_h)], fill="#EFEFEF")

# Header / toolbar area below status bar (keeps icons/text space clear)
toolbar_top = status_h
toolbar_bottom = 176
draw.rectangle([(0, toolbar_top), (1440, toolbar_bottom)], fill="#FFFFFF")

# Thin divider under toolbar
draw.line([(24, toolbar_bottom), (1440-24, toolbar_bottom)], fill="#E6E6E6", width=1)

# Event detail card (rounded) under the toolbar - background only
event_card_top = toolbar_bottom + 8
event_card_bottom = 420
event_card_bbox = (12, event_card_top, 1440-12, event_card_bottom)
draw.rounded_rectangle(event_card_bbox, radius=12, fill="#FFFFFF", outline="#EAEAEA", width=1)

# Separator under event details
draw.line([(24, event_card_bottom), (1440-24, event_card_bottom)], fill="#ECECEC", width=1)

# Big performers/content card area (white) that spans most of the scrollable area
performers_top = 640
performers_bottom = 2920
card_left = 12
card_right = 1440-12

# subtle shadow (drawn as a faint band beneath to imply elevation)
shadow_offset = 6
draw.rectangle([(card_left, performers_top+shadow_offset), (card_right, performers_top+shadow_offset+3)], fill="#F1F1F1")

# main performers card
draw.rectangle([(card_left, performers_top), (card_right, performers_bottom)], fill="#FFFFFF", outline="#E9E9E9", width=1)

# Section divider line above the "Performers" title area (subtle)
draw.line([(24, performers_top-48), (card_right-24, performers_top-48)], fill="#F2F2F2", width=1)

# Horizontal separators for rows in the performers list.
# Row tops detected in the layout; each row height is ~179. Draw separators at row bottoms.
row_tops = [745, 924, 1103, 1282, 1461, 1640, 1819, 1998, 2177, 2356, 2535, 2714]
for top in row_tops:
    bottom_y = top + 179
    # keep separators slightly inset from edges
    draw.line([(24, bottom_y), (card_right-24, bottom_y)], fill="#F4F4F4", width=1)

# Light vertical guideline on the left content margin to visually separate avatars from text (very subtle)
left_inset_x = 110  # leave space for circular avatars that will be pasted
draw.line([(left_inset_x, performers_top), (left_inset_x, performers_bottom)], fill="#FBFBFB", width=1)

# Final subtle bottom edge of content area
draw.line([(24, performers_bottom-24), (card_right-24, performers_bottom-24)], fill="#F2F2F2", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/00_icon_Keep_the_Party_Going_A_Tribute_to_Jimmy_.png
try:
    _c0 = get_crop(0, 1416, 179)
    canvas.paste(_c0, (12, 745), _c0)
except Exception:
    pass
layout["Keep_the_Party_Going:_A_T"] = [12, 745, 1428, 924]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/01_icon_7_events.png
try:
    _c1 = get_crop(1, 1416, 179)
    canvas.paste(_c1, (12, 2714), _c1)
except Exception:
    pass
layout["7_events"] = [12, 2714, 1428, 2893]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/02_icon_41_events.png
try:
    _c2 = get_crop(2, 1416, 179)
    canvas.paste(_c2, (12, 2535), _c2)
except Exception:
    pass
layout["41_events"] = [12, 2535, 1428, 2714]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/03_icon_my.png
try:
    _c3 = get_crop(3, 59, 60)
    canvas.paste(_c3, (112, 4), _c3)
except Exception:
    pass
layout["my"] = [112, 4, 171, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/04_icon_my.png
try:
    _c4 = get_crop(4, 61, 59)
    canvas.paste(_c4, (178, 4), _c4)
except Exception:
    pass
layout["my"] = [178, 4, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/05_icon_Performers.png
try:
    _c5 = get_crop(5, 1416, 179)
    canvas.paste(_c5, (12, 745), _c5)
except Exception:
    pass
layout["Performers"] = [12, 745, 1428, 924]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/06_icon_Kenny_Chesney.png
try:
    _c6 = get_crop(6, 1416, 179)
    canvas.paste(_c6, (12, 1103), _c6)
except Exception:
    pass
layout["Kenny_Chesney"] = [12, 1103, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/07_icon_Keep_the_Party_Going_A_Tribute_to_Jimmy_.png
try:
    _c7 = get_crop(7, 1416, 179)
    canvas.paste(_c7, (12, 924), _c7)
except Exception:
    pass
layout["Keep_the_Party_Going:_A_T"] = [12, 924, 1428, 1103]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 51, 66)
    canvas.paste(_c8, (1153, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1153, 1, 1204, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/09_icon_my.png
try:
    _c9 = get_crop(9, 55, 58)
    canvas.paste(_c9, (245, 5), _c9)
except Exception:
    pass
layout["my"] = [245, 5, 300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/10_icon_8.31.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (24, 84), _c10)
except Exception:
    pass
layout["8.31"] = [24, 84, 168, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 48, 57)
    canvas.paste(_c11, (1321, 4), _c11)
except Exception:
    pass
layout["icon_11"] = [1321, 4, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 73, 64)
    canvas.paste(_c12, (1213, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [1213, 1, 1286, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 53, 56)
    canvas.paste(_c13, (315, 5), _c13)
except Exception:
    pass
layout["icon_13"] = [315, 5, 368, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/14_icon_12_events.png
try:
    _c14 = get_crop(14, 1416, 179)
    canvas.paste(_c14, (12, 1282), _c14)
except Exception:
    pass
layout["12_events"] = [12, 1282, 1428, 1461]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/15_icon_21events.png
try:
    _c15 = get_crop(15, 1416, 179)
    canvas.paste(_c15, (12, 924), _c15)
except Exception:
    pass
layout["21events"] = [12, 924, 1428, 1103]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/16_icon_35_events.png
try:
    _c16 = get_crop(16, 1416, 179)
    canvas.paste(_c16, (12, 1103), _c16)
except Exception:
    pass
layout["35_events"] = [12, 1103, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/17_icon_Brandi_Carlile.png
try:
    _c17 = get_crop(17, 1416, 179)
    canvas.paste(_c17, (12, 1282), _c17)
except Exception:
    pass
layout["Brandi_Carlile"] = [12, 1282, 1428, 1461]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 49, 56)
    canvas.paste(_c18, (382, 5), _c18)
except Exception:
    pass
layout["icon_18"] = [382, 5, 431, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/19_icon_Zac_Brown_Band.png
try:
    _c19 = get_crop(19, 355, 52)
    canvas.paste(_c19, (244, 2213), _c19)
except Exception:
    pass
layout["Zac_Brown_Band"] = [244, 2213, 599, 2265]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/20_icon_30_events.png
try:
    _c20 = get_crop(20, 1416, 179)
    canvas.paste(_c20, (12, 2177), _c20)
except Exception:
    pass
layout["30_events"] = [12, 2177, 1428, 2356]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/21_icon_Jake_Shimabukuro.png
try:
    _c21 = get_crop(21, 1416, 179)
    canvas.paste(_c21, (12, 2535), _c21)
except Exception:
    pass
layout["Jake_Shimabukuro"] = [12, 2535, 1428, 2714]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/22_icon_Kenny_Chesney.png
try:
    _c22 = get_crop(22, 335, 53)
    canvas.paste(_c22, (244, 1138), _c22)
except Exception:
    pass
layout["Kenny_Chesney"] = [244, 1138, 579, 1191]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/23_icon_1_event.png
try:
    _c23 = get_crop(23, 1416, 179)
    canvas.paste(_c23, (12, 1640), _c23)
except Exception:
    pass
layout["1_event"] = [12, 1640, 1428, 1819]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/24_icon_1_event.png
try:
    _c24 = get_crop(24, 1416, 179)
    canvas.paste(_c24, (12, 1640), _c24)
except Exception:
    pass
layout["1_event"] = [12, 1640, 1428, 1819]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/25_icon_Jack_Johnson.png
try:
    _c25 = get_crop(25, 1416, 179)
    canvas.paste(_c25, (12, 2356), _c25)
except Exception:
    pass
layout["Jack_Johnson"] = [12, 2356, 1428, 2535]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/26_icon_1_event.png
try:
    _c26 = get_crop(26, 1416, 179)
    canvas.paste(_c26, (12, 2356), _c26)
except Exception:
    pass
layout["1_event"] = [12, 2356, 1428, 2535]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/27_icon_Jake_Shimabukuro.png
try:
    _c27 = get_crop(27, 1416, 179)
    canvas.paste(_c27, (12, 2714), _c27)
except Exception:
    pass
layout["Jake_Shimabukuro"] = [12, 2714, 1428, 2893]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 48, 62)
    canvas.paste(_c28, (1269, 2), _c28)
except Exception:
    pass
layout["icon_28"] = [1269, 2, 1317, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/29_icon_30_events.png
try:
    _c29 = get_crop(29, 1416, 179)
    canvas.paste(_c29, (12, 2177), _c29)
except Exception:
    pass
layout["30_events"] = [12, 2177, 1428, 2356]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/30_icon_2_events.png
try:
    _c30 = get_crop(30, 1416, 179)
    canvas.paste(_c30, (12, 1998), _c30)
except Exception:
    pass
layout["2_events"] = [12, 1998, 1428, 2177]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/31_icon_event.png
try:
    _c31 = get_crop(31, 1416, 179)
    canvas.paste(_c31, (12, 1819), _c31)
except Exception:
    pass
layout["event"] = [12, 1819, 1428, 1998]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/32_icon_Brandi_Carlile.png
try:
    _c32 = get_crop(32, 295, 54)
    canvas.paste(_c32, (245, 1316), _c32)
except Exception:
    pass
layout["Brandi_Carlile"] = [245, 1316, 540, 1370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/33_icon_Jackson_Browne.png
try:
    _c33 = get_crop(33, 1416, 179)
    canvas.paste(_c33, (12, 1461), _c33)
except Exception:
    pass
layout["Jackson_Browne"] = [12, 1461, 1428, 1640]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/34_icon_Jake_Shimabukuro.png
try:
    _c34 = get_crop(34, 1416, 179)
    canvas.paste(_c34, (12, 2714), _c34)
except Exception:
    pass
layout["Jake_Shimabukuro"] = [12, 2714, 1428, 2893]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/35_icon_Sheryl_Crow.png
try:
    _c35 = get_crop(35, 259, 54)
    canvas.paste(_c35, (247, 958), _c35)
except Exception:
    pass
layout["Sheryl_Crow"] = [247, 958, 506, 1012]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/36_icon_9_events.png
try:
    _c36 = get_crop(36, 1416, 179)
    canvas.paste(_c36, (12, 1461), _c36)
except Exception:
    pass
layout["9_events"] = [12, 1461, 1428, 1640]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/37_icon_Pitbull.png
try:
    _c37 = get_crop(37, 1416, 179)
    canvas.paste(_c37, (12, 1998), _c37)
except Exception:
    pass
layout["Pitbull"] = [12, 1998, 1428, 2177]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/38_icon_event.png
try:
    _c38 = get_crop(38, 1416, 179)
    canvas.paste(_c38, (12, 1819), _c38)
except Exception:
    pass
layout["event"] = [12, 1819, 1428, 1998]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/39_icon_Pitbull.png
try:
    _c39 = get_crop(39, 146, 52)
    canvas.paste(_c39, (244, 2034), _c39)
except Exception:
    pass
layout["Pitbull"] = [244, 2034, 390, 2086]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/40_text_8.31.png
try:
    _c40 = get_crop(40, 87, 43)
    canvas.paste(_c40, (20, 17), _c40)
except Exception:
    pass
layout["8.31"] = [20, 17, 107, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/41_text_Keep_the_Party_Going_A_Tribute_to.png
try:
    _c41 = get_crop(41, 1440, 113)
    canvas.paste(_c41, (0, 306), _c41)
except Exception:
    pass
layout["Keep_the_Party_Going:_A_T"] = [0, 306, 1440, 419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/42_text_Thu.png
try:
    _c42 = get_crop(42, 91, 45)
    canvas.paste(_c42, (211, 175), _c42)
except Exception:
    pass
layout["Thu,"] = [211, 175, 302, 220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/43_text_11_7_PM.png
try:
    _c43 = get_crop(43, 149, 43)
    canvas.paste(_c43, (378, 177), _c43)
except Exception:
    pass
layout["11,7_PM"] = [378, 177, 527, 220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/44_text_9.png
try:
    _c44 = get_crop(44, 117, 25)
    canvas.paste(_c44, (192, 233), _c44)
except Exception:
    pass
layout["9*"] = [192, 233, 309, 258]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/45_text_Get_directions.png
try:
    _c45 = get_crop(45, 1440, 113)
    canvas.paste(_c45, (0, 306), _c45)
except Exception:
    pass
layout["Get_directions"] = [0, 306, 1440, 419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/46_text_More_events_at_Hollywood_Bowl.png
try:
    _c46 = get_crop(46, 1440, 113)
    canvas.paste(_c46, (0, 419), _c46)
except Exception:
    pass
layout["More_events_at_Hollywood_"] = [0, 419, 1440, 532]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/47_text_Performers.png
try:
    _c47 = get_crop(47, 255, 52)
    canvas.paste(_c47, (56, 645), _c47)
except Exception:
    pass
layout["Performers"] = [56, 645, 311, 697]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_11_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-14/48_text_Caclcc.png
try:
    _c48 = get_crop(48, 140, 25)
    canvas.paste(_c48, (252, 2934), _c48)
except Exception:
    pass
layout["Caclcc"] = [252, 2934, 392, 2959]
