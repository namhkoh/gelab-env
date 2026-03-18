# page_id: page_seatgeek_6623dffea11a48f2955bafde23a3f1c7_05
# screenshot: 2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8.png
# step_index: 5/9
# task: Open SeatGeek. Search "New York Knicks" and select the second upcoming event, show the location of the event and track the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top background (deep navy) and status bar
draw.rectangle((0, 0, 1440, 520), fill=(6, 50, 70))          # main header background
draw.rectangle((0, 0, 90, 1440), fill=(4, 36, 52))          # slightly darker status bar strip (left large to ensure full coverage)
# subtle darker strip for exact status bar band
draw.rectangle((0, 0, 1440, 90), fill=(4, 36, 52))

# Decorative soft shapes in header (subtle, non-icon background shapes)
# large rounded ticket-like shape top-right
draw.rounded_rectangle((980, -40, 1500, 360), radius=60, fill=(7, 63, 86))
# plus-like soft shape in center
draw.ellipse((590, 150, 690, 250), fill=(9, 68, 92))

# White content panel with rounded top corners overlapping header
panel_top = 520
draw.rounded_rectangle((0, panel_top, 1440, 2960), radius=28, fill=(255, 255, 255))

# subtle top shadow line to separate header from content
draw.line((24, panel_top, 1440-24, panel_top), fill=(220, 224, 226), width=2)

# "Protected by our Buyer Guarantee" area divider (background already white, add faint divider below)
buyer_divider_y = 933 + 126  # bottom of detected buyer guarantee area
draw.line((24, buyer_divider_y, 1440-24, buyer_divider_y), fill=(235, 238, 240), width=1)

# Event list card backgrounds (rounded cards behind each detected event crop)
card_margin_x = 48
card_width = 1440 - card_margin_x*2
# Detected event blocks at y positions: 1279, 1646, 2013, 2380 (each size 367)
event_tops = [1279, 1646, 2013, 2380]
card_inset = 14
for top in event_tops:
    top_y = top + card_inset
    bottom_y = top + 367 - card_inset
    # subtle shadow rectangle behind card
    shadow_offset = 8
    draw.rounded_rectangle(
        (card_margin_x+shadow_offset, top_y+shadow_offset, card_margin_x+card_width+shadow_offset, bottom_y+shadow_offset),
        radius=18, fill=(245, 247, 248)
    )
    # main card
    draw.rounded_rectangle((card_margin_x, top_y, card_margin_x+card_width, bottom_y), radius=18, fill=(255, 255, 255))
    # faint inner divider at card bottom
    draw.line((card_margin_x+20, bottom_y, card_margin_x+card_width-20, bottom_y), fill=(240, 242, 243), width=1)

# Thin separators between event blocks (subtle)
for sep_y in [1279-12, 1646-12, 2013-12, 2380-12, 2747]:
    draw.line((24, sep_y, 1440-24, sep_y), fill=(245, 246, 247), width=1)

# "All Games" section area top separator (around detected text at ~2791)
all_games_sep_y = 2720
draw.line((24, all_games_sep_y, 1440-24, all_games_sep_y), fill=(230, 233, 235), width=2)

# Bottom safe area subtle fade (to anchor the page)
draw.rectangle((0, 2880, 1440, 2960), fill=(249, 250, 251))

# small left content column background (to suggest the date badges column, keep pale and non-specific)
# These are only background panels; icons/text for dates will be pasted on top in detected positions.
date_col_x = 36
date_col_w = 180
for top in event_tops:
    draw.rounded_rectangle((date_col_x, top+28, date_col_x+date_col_w, top+327), radius=16, fill=(250, 251, 251))

# final faint vertical margins to frame content
draw.rectangle((0, panel_top, 24, 2960), fill=(255,255,255))
draw.rectangle((1440-24, panel_top, 1440, 2960), fill=(255,255,255))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/00_icon_Madison_Square_Garden.png
try:
    _c0 = get_crop(0, 1440, 367)
    canvas.paste(_c0, (0, 1646), _c0)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 1646, 1440, 2013]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/01_icon_04.png
try:
    _c1 = get_crop(1, 1440, 367)
    canvas.paste(_c1, (0, 2013), _c1)
except Exception:
    pass
layout["04"] = [0, 2013, 1440, 2380]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/02_icon_76ers_at_New_York_Knicks_Game_7.png
try:
    _c2 = get_crop(2, 1440, 367)
    canvas.paste(_c2, (0, 2013), _c2)
except Exception:
    pass
layout["76ers_at_New_York_Knicks_"] = [0, 2013, 1440, 2380]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/03_icon_Track_this_performer.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1104, 84), _c3)
except Exception:
    pass
layout["Track_this_performer"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/04_icon_Share_this_performer.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1260, 84), _c4)
except Exception:
    pass
layout["Share_this_performer"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/05_icon_30.png
try:
    _c5 = get_crop(5, 1440, 367)
    canvas.paste(_c5, (0, 1646), _c5)
except Exception:
    pass
layout["30"] = [0, 1646, 1440, 2013]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/06_icon_Madison_Square_Garden.png
try:
    _c6 = get_crop(6, 1440, 367)
    canvas.paste(_c6, (0, 1279), _c6)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 1279, 1440, 1646]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/07_icon_22.png
try:
    _c7 = get_crop(7, 1440, 367)
    canvas.paste(_c7, (0, 1279), _c7)
except Exception:
    pass
layout["22"] = [0, 1279, 1440, 1646]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/08_icon_TBD.png
try:
    _c8 = get_crop(8, 192, 228)
    canvas.paste(_c8, (66, 2404), _c8)
except Exception:
    pass
layout["TBD"] = [66, 2404, 258, 2632]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/09_icon_6.57_W.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (36, 84), _c9)
except Exception:
    pass
layout["6.57_W"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 61, 63)
    canvas.paste(_c10, (243, 4), _c10)
except Exception:
    pass
layout["icon_10"] = [243, 4, 304, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/11_icon_6.57_W.png
try:
    _c11 = get_crop(11, 56, 62)
    canvas.paste(_c11, (180, 3), _c11)
except Exception:
    pass
layout["6.57_W"] = [180, 3, 236, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/12_icon_6.57_W.png
try:
    _c12 = get_crop(12, 60, 64)
    canvas.paste(_c12, (114, 2), _c12)
except Exception:
    pass
layout["6.57_W"] = [114, 2, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 58, 71)
    canvas.paste(_c13, (1149, 2), _c13)
except Exception:
    pass
layout["icon_13"] = [1149, 2, 1207, 73]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 59, 75)
    canvas.paste(_c14, (380, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [380, 0, 439, 75]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/15_icon_Protected_by_our_Buyer_Guarantee.png
try:
    _c15 = get_crop(15, 1440, 126)
    canvas.paste(_c15, (0, 933), _c15)
except Exception:
    pass
layout["Protected_by_our_Buyer_Gu"] = [0, 933, 1440, 1059]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 63, 68)
    canvas.paste(_c16, (312, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [312, 1, 375, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 95, 115)
    canvas.paste(_c17, (1299, 950), _c17)
except Exception:
    pass
layout["icon_17"] = [1299, 950, 1394, 1065]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/18_icon_Madison_Square_Garden.png
try:
    _c18 = get_crop(18, 1440, 367)
    canvas.paste(_c18, (0, 2380), _c18)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 2380, 1440, 2747]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 59, 67)
    canvas.paste(_c19, (1215, 3), _c19)
except Exception:
    pass
layout["icon_19"] = [1215, 3, 1274, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 48, 65)
    canvas.paste(_c20, (1323, 4), _c20)
except Exception:
    pass
layout["icon_20"] = [1323, 4, 1371, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/21_icon_Eastern_Conference_First_Round_Philadelp.png
try:
    _c21 = get_crop(21, 1440, 367)
    canvas.paste(_c21, (0, 1646), _c21)
except Exception:
    pass
layout["Eastern_Conference_First_"] = [0, 1646, 1440, 2013]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/22_icon_Eastern_Conference_First_Round_Philadelp.png
try:
    _c22 = get_crop(22, 1440, 367)
    canvas.paste(_c22, (0, 2013), _c22)
except Exception:
    pass
layout["Eastern_Conference_First_"] = [0, 2013, 1440, 2380]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/23_text_New_York_NY.png
try:
    _c23 = get_crop(23, 352, 62)
    canvas.paste(_c23, (55, 1177), _c23)
except Exception:
    pass
layout["New_York,_NY"] = [55, 1177, 407, 1239]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/24_text_AIl_Games.png
try:
    _c24 = get_crop(24, 265, 55)
    canvas.paste(_c24, (60, 2791), _c24)
except Exception:
    pass
layout["AIl_Games"] = [60, 2791, 325, 2846]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/25_text_Factorn.png
try:
    _c25 = get_crop(25, 177, 27)
    canvas.paste(_c25, (317, 2932), _c25)
except Exception:
    pass
layout["Factorn"] = [317, 2932, 494, 2959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/26_text_Conforonco.png
try:
    _c26 = get_crop(26, 267, 27)
    canvas.paste(_c26, (502, 2932), _c26)
except Exception:
    pass
layout["Conforonco"] = [502, 2932, 769, 2959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_05_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-8/27_text_Firct_Pound_Philadelnbia.png
try:
    _c27 = get_crop(27, 564, 29)
    canvas.paste(_c27, (779, 2930), _c27)
except Exception:
    pass
layout["Firct_Pound:Philadelnbia"] = [779, 2930, 1343, 2959]
