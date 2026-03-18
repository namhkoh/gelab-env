# page_id: page_seatgeek_49ceba1342134bb89f14fac27abc2dcd_06
# screenshot: 2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9.png
# step_index: 6/12
# task: Open SeatGeek. Track "New York Yankees", "Boston Red Sox".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (canvas: PIL.Image, draw: PIL.ImageDraw)
w, h = canvas.size

# Colors
bg_color = (250, 251, 252)         # very light neutral background
status_color = (20, 26, 33)        # dark status bar tone
navy_stripe = (19, 40, 77)         # deep navy accent stripe
stripe_edge = (31, 49, 79)         # slightly lighter edge for stripe
separator = (230, 232, 235)        # light grey separators
soft_shadow = (240, 241, 242)      # subtle shadow/raised effect

# Fill overall background (match app's off-white)
draw.rectangle([0, 0, w, h], fill=bg_color)

# Status bar area at top (~0-110px)
status_h = 110
draw.rectangle([0, 0, w, status_h], fill=status_color)

# Thin translucent-ish overlay band below status (mimic subtle fade under top image)
draw.rectangle([0, status_h-10, w, status_h+14], fill=(28, 36, 48))

# Navy separator stripe under hero/banner image (full-width accent bar)
stripe_top = 480
stripe_bottom = 516
draw.rectangle([0, stripe_top, w, stripe_bottom], fill=navy_stripe)
# add a thin lighter edge to give separation
draw.rectangle([0, stripe_bottom, w, stripe_bottom+3], fill=stripe_edge)

# Subtle divider under the header info block (below "Protected by our Buyer Guarantee" area)
# Text block bottom ~1059 per detected element; draw divider close below it
header_div_y = 1064
draw.line([(24, header_div_y), (w-24, header_div_y)], fill=separator, width=1)

# Slight raised band to indicate content container start
draw.rectangle([20, header_div_y+8, w-20, header_div_y+12], fill=soft_shadow)

# Section separators aligned with event list groups (using detected top positions minus offset)
event_tops = [1279, 1572, 1865, 2303, 2596]  # detected element top Ys
for top in event_tops:
    sep_y = top - 16
    # draw a faint full-width separator with left/right insets
    draw.line([(32, sep_y), (w-32, sep_y)], fill=separator, width=1)

# Additional subtle separators between major blocks
extra_seps = [720, 920, 1150]  # visual separators across the page
for y in extra_seps:
    draw.line([(32, y), (w-32, y)], fill=(245, 246, 247), width=1)

# Large rounded container behind the "All Games" / list area (subtle, do not overlap individual cards)
list_container_top = 1200
list_container_margin = 24
draw.rounded_rectangle(
    [list_container_margin, list_container_top, w-list_container_margin, h-60],
    radius=16,
    fill=(255, 255, 255),
    outline=None
)

# Soft top shadow for the list container to separate it from header
shadow_y = list_container_top - 6
draw.rectangle([list_container_margin+4, shadow_y, w-list_container_margin-4, shadow_y+4], fill=soft_shadow)

# Left and right page gutters to suggest app margins
gutters_color = bg_color
draw.rectangle([0, 0, 24, h], fill=gutters_color)
draw.rectangle([w-24, 0, w, h], fill=gutters_color)

# Bottom divider for footer separation
footer_div_y = h - 220
draw.line([(24, footer_div_y), (w-24, footer_div_y)], fill=separator, width=1)

# End of decorative/background-only drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/00_icon_03.png
try:
    _c0 = get_crop(0, 1440, 293)
    canvas.paste(_c0, (0, 1572), _c0)
except Exception:
    pass
layout["03"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/01_icon_22.png
try:
    _c1 = get_crop(1, 1440, 293)
    canvas.paste(_c1, (0, 2303), _c1)
except Exception:
    pass
layout["22"] = [0, 2303, 1440, 2596]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/02_icon_02.png
try:
    _c2 = get_crop(2, 1440, 293)
    canvas.paste(_c2, (0, 1279), _c2)
except Exception:
    pass
layout["02"] = [0, 1279, 1440, 1572]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/03_icon_04.png
try:
    _c3 = get_crop(3, 1440, 293)
    canvas.paste(_c3, (0, 1865), _c3)
except Exception:
    pass
layout["04"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/04_icon_23.png
try:
    _c4 = get_crop(4, 1440, 293)
    canvas.paste(_c4, (0, 2596), _c4)
except Exception:
    pass
layout["23"] = [0, 2596, 1440, 2889]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/05_icon_8.35.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (36, 84), _c5)
except Exception:
    pass
layout["8.35"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/06_icon_ISTARRIBSF.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1260, 84), _c6)
except Exception:
    pass
layout["ISTARRIBSF"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/07_icon_New_York_Yankees.png
try:
    _c7 = get_crop(7, 203, 199)
    canvas.paste(_c7, (51, 606), _c7)
except Exception:
    pass
layout["New_York_Yankees"] = [51, 606, 254, 805]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/08_icon_Globe_Life_Field.png
try:
    _c8 = get_crop(8, 1440, 293)
    canvas.paste(_c8, (0, 1279), _c8)
except Exception:
    pass
layout["Globe_Life_Field"] = [0, 1279, 1440, 1572]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/09_icon_ISTARRIBSF.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1104, 84), _c9)
except Exception:
    pass
layout["ISTARRIBSF"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/10_icon_8.35.png
try:
    _c10 = get_crop(10, 138, 84)
    canvas.paste(_c10, (0, 0), _c10)
except Exception:
    pass
layout["8.35"] = [0, 0, 138, 84]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/11_icon_Globe_Life_Field.png
try:
    _c11 = get_crop(11, 1440, 293)
    canvas.paste(_c11, (0, 1572), _c11)
except Exception:
    pass
layout["Globe_Life_Field"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 81, 96)
    canvas.paste(_c12, (1306, 960), _c12)
except Exception:
    pass
layout["icon_12"] = [1306, 960, 1387, 1056]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/13_icon_YAMKEES.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1104, 84), _c13)
except Exception:
    pass
layout["YAMKEES"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/14_icon_New_York_Yankees_at_Texas_Rangers.png
try:
    _c14 = get_crop(14, 1440, 293)
    canvas.paste(_c14, (0, 1572), _c14)
except Exception:
    pass
layout["New_York_Yankees_at_Texas"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/15_icon_Oakland_Athletics_at_New_York_Yankees.png
try:
    _c15 = get_crop(15, 1440, 293)
    canvas.paste(_c15, (0, 2303), _c15)
except Exception:
    pass
layout["Oakland_Athletics_at_New_"] = [0, 2303, 1440, 2596]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/16_icon_New_York_Yankees_at_Texas_Rangers.png
try:
    _c16 = get_crop(16, 1440, 293)
    canvas.paste(_c16, (0, 1865), _c16)
except Exception:
    pass
layout["New_York_Yankees_at_Texas"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/17_icon_Oakland_Athletics_at_New_York_Yankees.png
try:
    _c17 = get_crop(17, 1440, 293)
    canvas.paste(_c17, (0, 2596), _c17)
except Exception:
    pass
layout["Oakland_Athletics_at_New_"] = [0, 2596, 1440, 2889]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/18_icon_S1_Hot_Dog.png
try:
    _c18 = get_crop(18, 1440, 293)
    canvas.paste(_c18, (0, 1865), _c18)
except Exception:
    pass
layout["S1_Hot_Dog"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/19_icon_ISTARRIBSF.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1104, 84), _c19)
except Exception:
    pass
layout["ISTARRIBSF"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/20_text_New_York_Yankees.png
try:
    _c20 = get_crop(20, 526, 64)
    canvas.paste(_c20, (59, 859), _c20)
except Exception:
    pass
layout["New_York_Yankees"] = [59, 859, 585, 923]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/21_text_Protected_by_our_Buyer_Guarantee.png
try:
    _c21 = get_crop(21, 1440, 126)
    canvas.paste(_c21, (0, 933), _c21)
except Exception:
    pass
layout["Protected_by_our_Buyer_Gu"] = [0, 933, 1440, 1059]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/22_text_Dallas_TX.png
try:
    _c22 = get_crop(22, 271, 69)
    canvas.paste(_c22, (53, 1174), _c22)
except Exception:
    pass
layout["Dallas,_TX"] = [53, 1174, 324, 1243]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/23_text_Oakland.png
try:
    _c23 = get_crop(23, 200, 32)
    canvas.paste(_c23, (317, 2927), _c23)
except Exception:
    pass
layout["Oakland"] = [317, 2927, 517, 2959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/24_text_Athleticc_at.png
try:
    _c24 = get_crop(24, 263, 32)
    canvas.paste(_c24, (527, 2927), _c24)
except Exception:
    pass
layout["Athleticc_at"] = [527, 2927, 790, 2959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/25_text_New.png
try:
    _c25 = get_crop(25, 96, 29)
    canvas.paste(_c25, (800, 2930), _c25)
except Exception:
    pass
layout["New"] = [800, 2930, 896, 2959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_06_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-9/26_text_Vork_Vankeec.png
try:
    _c26 = get_crop(26, 303, 32)
    canvas.paste(_c26, (906, 2927), _c26)
except Exception:
    pass
layout["Vork_Vankeec"] = [906, 2927, 1209, 2959]
