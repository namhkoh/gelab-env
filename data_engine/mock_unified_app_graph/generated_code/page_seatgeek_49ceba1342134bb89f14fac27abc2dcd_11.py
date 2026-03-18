# page_id: page_seatgeek_49ceba1342134bb89f14fac27abc2dcd_11
# screenshot: 2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14.png
# step_index: 11/12
# task: Open SeatGeek. Track "New York Yankees", "Boston Red Sox".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided. Fonts: font_sm, font_md, font_lg, font_xl

# Colors
STATUS_BAR_COLOR = (8, 58, 107)        # deep blue for status bar
HERO_TOP_COLOR = (44, 120, 191)        # stadium-photo placeholder (lighter blue)
HERO_BOTTOM_COLOR = (10, 36, 64)       # darker trim under hero
CONTENT_BG = (255, 255, 255)           # main content white
CARD_BG = (250, 250, 250)              # subtle off-white card background
DIVIDER = (230, 230, 230)              # light grey dividers
SHADOW = (220, 220, 220)               # subtle shadow color

W, H = canvas.size

# 1) Status bar area at the very top (~96px)
status_h = 96
draw.rectangle([0, 0, W, status_h], fill=STATUS_BAR_COLOR)

# 2) Hero image placeholder area (beneath status bar)
hero_top = status_h
hero_bottom = 460
# simple vertical two-band to suggest image area (do not draw any icons/text)
draw.rectangle([0, hero_top, W, hero_bottom], fill=HERO_TOP_COLOR)
# darker gradient band at bottom edge of hero
draw.rectangle([0, hero_bottom-28, W, hero_bottom+8], fill=HERO_BOTTOM_COLOR)

# 3) Thin divider line below the hero image
draw.line([(40, hero_bottom+12), (W-40, hero_bottom+12)], fill=DIVIDER, width=2)

# 4) Main content background (white) from just below hero to bottom
content_top = hero_bottom + 12
draw.rectangle([0, content_top, W, H], fill=CONTENT_BG)

# 5) Horizontal navy trim behind team/logo area (stretching slightly into content)
trim_y1 = hero_bottom - 20
trim_y2 = hero_bottom + 8
draw.rectangle([0, trim_y1, W, trim_y2], fill=HERO_BOTTOM_COLOR)

# 6) Primary info card (rounded rectangle) under the hero for title / badge background
card_x1, card_x2 = 48, W - 48
card_y1, card_y2 = content_top + 28, content_top + 220
radius = 20
# subtle shadow band behind card (simple rectangle, slightly offset)
draw.rectangle([card_x1+6, card_y1+6, card_x2+6, card_y2+6], fill=SHADOW)
# card background
try:
    draw.rounded_rectangle([card_x1, card_y1, card_x2, card_y2], radius=radius, fill=CARD_BG, outline=DIVIDER, width=1)
except Exception:
    # fallback if rounded_rectangle not supported
    draw.rectangle([card_x1, card_y1, card_x2, card_y2], fill=CARD_BG, outline=DIVIDER)

# 7) Section header background band for "Dallas, TX" / first list group (subtle)
section1_y = card_y2 + 34
section1_h = 96
draw.rectangle([0, section1_y, W, section1_y + section1_h], fill=CONTENT_BG)
# small divider above this section
draw.line([(40, section1_y), (W-40, section1_y)], fill=DIVIDER, width=1)

# 8) Draw light rounded cards as backgrounds for the repeating game list groups.
# We'll place a few repeated rounded rect backgrounds spaced vertically — these are only background shapes.
row_x1, row_x2 = 36, W - 36
row_w = row_x2 - row_x1
row_h = 170
row_radius = 18
start_y = section1_y + 18
gap = 28

for i in range(6):
    y1 = start_y + i * (row_h + gap)
    y2 = y1 + row_h
    # subtle shadow
    draw.rectangle([row_x1+4, y1+6, row_x2+4, y2+6], fill=(240,240,240))
    # card background (very light)
    try:
        draw.rounded_rectangle([row_x1, y1, row_x2, y2], radius=row_radius, fill=CONTENT_BG, outline=(245,245,245), width=1)
    except Exception:
        draw.rectangle([row_x1, y1, row_x2, y2], fill=CONTENT_BG, outline=(245,245,245))

# 9) Section separators between major groups (thin full-width lines)
separator_ys = [
    card_y2 + 18,        # below primary card
    start_y - 10,        # before first list item group
    start_y + 3*(row_h+gap) - 10,  # middle separator
    start_y + 6*(row_h+gap) + 24   # lower content break
]
for sy in separator_ys:
    if 0 < sy < H:
        draw.line([(36, sy), (W-36, sy)], fill=DIVIDER, width=1)

# 10) Light footer/top-of-bottom-bar area hint (do not draw icons)
footer_hint_h = 220
draw.rectangle([0, H-footer_hint_h, W, H], fill=CONTENT_BG)
# subtle top divider for footer hint
draw.line([(36, H-footer_hint_h), (W-36, H-footer_hint_h)], fill=DIVIDER, width=1)

# Note: All actual icons, text, and buttons will be pasted later on top of these backgrounds.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/00_icon_24.png
try:
    _c0 = get_crop(0, 1440, 293)
    canvas.paste(_c0, (0, 2596), _c0)
except Exception:
    pass
layout["24"] = [0, 2596, 1440, 2889]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/01_icon_38.png
try:
    _c1 = get_crop(1, 204, 201)
    canvas.paste(_c1, (51, 602), _c1)
except Exception:
    pass
layout["38"] = [51, 602, 255, 803]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/02_icon_04.png
try:
    _c2 = get_crop(2, 1440, 293)
    canvas.paste(_c2, (0, 1865), _c2)
except Exception:
    pass
layout["04"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/03_icon_03.png
try:
    _c3 = get_crop(3, 1440, 293)
    canvas.paste(_c3, (0, 1572), _c3)
except Exception:
    pass
layout["03"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/04_icon_23.png
try:
    _c4 = get_crop(4, 1440, 293)
    canvas.paste(_c4, (0, 2303), _c4)
except Exception:
    pass
layout["23"] = [0, 2303, 1440, 2596]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/05_icon_Share_this_performer.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1260, 84), _c5)
except Exception:
    pass
layout["Share_this_performer"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/06_icon_Track_this_performer.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1104, 84), _c6)
except Exception:
    pass
layout["Track_this_performer"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/07_icon_02.png
try:
    _c7 = get_crop(7, 1440, 293)
    canvas.paste(_c7, (0, 1279), _c7)
except Exception:
    pass
layout["02"] = [0, 1279, 1440, 1572]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/08_icon_8.35_Wy.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (36, 84), _c8)
except Exception:
    pass
layout["8.35_Wy"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/09_icon_Boston_Red_Sox_at_Cleveland_Guardians.png
try:
    _c9 = get_crop(9, 1440, 293)
    canvas.paste(_c9, (0, 2303), _c9)
except Exception:
    pass
layout["Boston_Red_Sox_at_Clevela"] = [0, 2303, 1440, 2596]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/10_icon_Globe_Life_Field.png
try:
    _c10 = get_crop(10, 1440, 293)
    canvas.paste(_c10, (0, 1279), _c10)
except Exception:
    pass
layout["Globe_Life_Field"] = [0, 1279, 1440, 1572]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/11_icon_Globe_Life_Field.png
try:
    _c11 = get_crop(11, 1440, 293)
    canvas.paste(_c11, (0, 1572), _c11)
except Exception:
    pass
layout["Globe_Life_Field"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/12_icon_Boston_Red_Sox_at_Cleveland_Guardians.png
try:
    _c12 = get_crop(12, 1440, 293)
    canvas.paste(_c12, (0, 2596), _c12)
except Exception:
    pass
layout["Boston_Red_Sox_at_Clevela"] = [0, 2596, 1440, 2889]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 84, 102)
    canvas.paste(_c13, (1305, 957), _c13)
except Exception:
    pass
layout["icon_13"] = [1305, 957, 1389, 1059]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 59, 75)
    canvas.paste(_c14, (1147, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1147, 0, 1206, 75]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/15_icon_Boston_Red_Sox_at_Texas_Rangers.png
try:
    _c15 = get_crop(15, 1440, 293)
    canvas.paste(_c15, (0, 1865), _c15)
except Exception:
    pass
layout["Boston_Red_Sox_at_Texas_R"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/16_icon_Boston_Red_Sox_at_Texas_Rangers.png
try:
    _c16 = get_crop(16, 1440, 293)
    canvas.paste(_c16, (0, 1572), _c16)
except Exception:
    pass
layout["Boston_Red_Sox_at_Texas_R"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 111, 74)
    canvas.paste(_c17, (1213, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1213, 0, 1324, 74]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/18_icon_Globe_Life_Field.png
try:
    _c18 = get_crop(18, 1440, 293)
    canvas.paste(_c18, (0, 1865), _c18)
except Exception:
    pass
layout["Globe_Life_Field"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 59, 74)
    canvas.paste(_c19, (1317, 0), _c19)
except Exception:
    pass
layout["icon_19"] = [1317, 0, 1376, 74]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/20_text_8.35_Wy.png
try:
    _c20 = get_crop(20, 153, 49)
    canvas.paste(_c20, (19, 12), _c20)
except Exception:
    pass
layout["8.35_Wy"] = [19, 12, 172, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/21_text_Boston_Red_Sox.png
try:
    _c21 = get_crop(21, 452, 64)
    canvas.paste(_c21, (57, 859), _c21)
except Exception:
    pass
layout["Boston_Red_Sox"] = [57, 859, 509, 923]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/22_text_Protected_by_our_Buyer_Guarantee.png
try:
    _c22 = get_crop(22, 1440, 126)
    canvas.paste(_c22, (0, 933), _c22)
except Exception:
    pass
layout["Protected_by_our_Buyer_Gu"] = [0, 933, 1440, 1059]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/23_text_Dallas_TX.png
try:
    _c23 = get_crop(23, 271, 69)
    canvas.paste(_c23, (53, 1174), _c23)
except Exception:
    pass
layout["Dallas,_TX"] = [53, 1174, 324, 1243]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/24_text_Rocton_Red_Goy_Ot_Cleveland.png
try:
    _c24 = get_crop(24, 1440, 293)
    canvas.paste(_c24, (0, 2596), _c24)
except Exception:
    pass
layout["Rocton_Red_Goy_Ot_Clevela"] = [0, 2596, 1440, 2889]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_11_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-14/25_text_Guardianc.png
try:
    _c25 = get_crop(25, 240, 29)
    canvas.paste(_c25, (973, 2930), _c25)
except Exception:
    pass
layout["Guardianc"] = [973, 2930, 1213, 2959]
