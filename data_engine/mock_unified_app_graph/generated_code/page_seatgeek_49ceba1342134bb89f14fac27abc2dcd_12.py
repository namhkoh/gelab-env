# page_id: page_seatgeek_49ceba1342134bb89f14fac27abc2dcd_12
# screenshot: 2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15.png
# step_index: 12/12
# task: Open SeatGeek. Track "New York Yankees", "Boston Red Sox".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for a 1440x2960 canvas.
# Available variables:
# - canvas: PIL Image (1440x2960 RGB)
# - draw: PIL ImageDraw.Draw(canvas)
# - font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Base background (slightly off-white to match app background)
draw.rectangle((0, 0, w, h), fill=(249, 250, 251))

# ---------- Status bar area (top) ----------
status_h = 88
# Solid dark overlay for status area
draw.rectangle((0, 0, w, status_h), fill=(7, 46, 78))

# subtle bottom divider under status
draw.line((0, status_h - 1, w, status_h - 1), fill=(255, 255, 255, 40), width=1)

# ---------- Hero / header image background ----------
hero_h = 440
# Sky gradient block (simple solid + subtle bands to suggest stadium)
draw.rectangle((0, status_h, w, hero_h), fill=(37, 132, 199))

# Crowd/stadium band (middle of hero) - a textured band made of horizontal stripes
crowd_top = status_h + 110
crowd_bottom = status_h + 240
for i in range(0, 20):
    # alternating warm/dark stripes to suggest audience/stands
    y0 = crowd_top + i * ((crowd_bottom - crowd_top) / 20)
    y1 = y0 + ((crowd_bottom - crowd_top) / 20) + 1
    color = (94, 42, 42) if i % 2 == 0 else (137, 65, 65)
    draw.rectangle((0, y0, w, y1), fill=color)

# Stadium field (green ellipse)
field_bbox = (-240, status_h + 80, w + 240, hero_h + 160)
draw.ellipse(field_bbox, fill=(88, 154, 62))

# Subtle vignette overlay at bottom of hero to separate from content
vignette_y = hero_h - 12
draw.rectangle((0, vignette_y, w, hero_h), fill=(11, 34, 57))

# Thin navy divider strip under hero
divider_h = hero_h
draw.rectangle((0, divider_h - 8, w, divider_h), fill=(6, 30, 56))

# ---------- Header card / team info area ----------
# White rounded card that sits partly overlapping the hero
card_left = 36
card_right = w - 36
card_top = divider_h - 36
card_bottom = card_top + 320
card_radius = 16
draw.rounded_rectangle(
    (card_left, card_top, card_right, card_bottom),
    radius=card_radius,
    fill=(255, 255, 255),
    outline=(230, 230, 230),
    width=1
)

# Thin separator line inside header card (to separate header title area from details)
sep_y = card_top + 120
draw.line((card_left + 12, sep_y, card_right - 12, sep_y), fill=(236, 238, 240), width=1)

# Small left accent bar under the hero to echo team badge area (abstract background only)
accent_w = 98
accent_h = 98
accent_x = card_left + 24
accent_y = card_top - 34
draw.rectangle((accent_x, accent_y, accent_x + accent_w, accent_y + accent_h), fill=(18, 61, 105), outline=(12, 45, 84))

# ---------- Main content area ----------
content_top = card_bottom + 28
# Large white content sheet
draw.rectangle((0, content_top, w, h), fill=(249, 250, 251))

# Section header divider "Dallas, TX" area - draw the section heading baseline and separator
section1_top = content_top + 34
section1_height = 80
draw.rectangle((24, section1_top, w - 24, section1_top + section1_height), fill=(249, 250, 251))
# subtle separator line under section title
draw.line((24, section1_top + section1_height, w - 24, section1_top + section1_height), fill=(233, 235, 237), width=1)

# ---------- List rows structure ----------
# We'll draw the right-hand text card backgrounds for each list item, leaving the left date-pill area clear
row_start_y = section1_top + section1_height + 24
row_height = 160
row_gap = 18
text_area_left = 200
text_area_right = w - 36

for i in range(6):
    y0 = int(row_start_y + i * (row_height + row_gap))
    y1 = int(y0 + row_height)
    # Background for the textual part of the row (white card with subtle shadow)
    draw.rounded_rectangle(
        (text_area_left, y0, text_area_right, y1),
        radius=14,
        fill=(255, 255, 255),
        outline=(236, 238, 240),
        width=1
    )
    # subtle bottom separator to distinguish rows
    draw.line((text_area_left + 12, y1 + 8, text_area_right - 12, y1 + 8), fill=(241, 242, 243), width=1)

# ---------- "All Games" section header ----------
all_games_y = row_start_y + 3 * (row_height + row_gap) + 28
draw.rectangle((24, all_games_y - 16, w - 24, all_games_y + 64), fill=(249, 250, 251))
# divider under "All Games"
draw.line((24, all_games_y + 64, w - 24, all_games_y + 64), fill=(233, 235, 237), width=1)

# ---------- Additional separators further down the list ----------
more_start = all_games_y + 96
for j in range(6):
    yy = more_start + j * (row_height + row_gap)
    # right-hand text area only
    draw.rounded_rectangle(
        (text_area_left, yy, text_area_right, yy + row_height),
        radius=14,
        fill=(255, 255, 255),
        outline=(236, 238, 240),
        width=1
    )
    draw.line((text_area_left + 12, yy + row_height + 8, text_area_right - 12, yy + row_height + 8), fill=(241, 242, 243), width=1)

# ---------- Footer safe area (bottom) ----------
footer_h = 140
draw.rectangle((0, h - footer_h, w, h), fill=(249, 250, 251))
# top divider for footer
draw.line((24, h - footer_h, w - 24, h - footer_h), fill=(233, 235, 237), width=1)

# Note: icons, badges, and all textual content will be pasted on top separately at exact detected positions.
# This drawing only provides the background, card surfaces, dividers, and structural shapes.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/00_icon_24.png
try:
    _c0 = get_crop(0, 1440, 293)
    canvas.paste(_c0, (0, 2596), _c0)
except Exception:
    pass
layout["24"] = [0, 2596, 1440, 2889]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/01_icon_38.png
try:
    _c1 = get_crop(1, 204, 201)
    canvas.paste(_c1, (51, 602), _c1)
except Exception:
    pass
layout["38"] = [51, 602, 255, 803]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/02_icon_04.png
try:
    _c2 = get_crop(2, 1440, 293)
    canvas.paste(_c2, (0, 1865), _c2)
except Exception:
    pass
layout["04"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/03_icon_03.png
try:
    _c3 = get_crop(3, 1440, 293)
    canvas.paste(_c3, (0, 1572), _c3)
except Exception:
    pass
layout["03"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/04_icon_23.png
try:
    _c4 = get_crop(4, 1440, 293)
    canvas.paste(_c4, (0, 2303), _c4)
except Exception:
    pass
layout["23"] = [0, 2303, 1440, 2596]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/05_icon_Track_this_performer.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1104, 84), _c5)
except Exception:
    pass
layout["Track_this_performer"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/06_icon_Share_this_performer.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1260, 84), _c6)
except Exception:
    pass
layout["Share_this_performer"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/07_icon_02.png
try:
    _c7 = get_crop(7, 1440, 293)
    canvas.paste(_c7, (0, 1279), _c7)
except Exception:
    pass
layout["02"] = [0, 1279, 1440, 1572]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/08_icon_8.35_Wy.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (36, 84), _c8)
except Exception:
    pass
layout["8.35_Wy"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/09_icon_Boston_Red_Sox_at_Cleveland_Guardians.png
try:
    _c9 = get_crop(9, 1440, 293)
    canvas.paste(_c9, (0, 2303), _c9)
except Exception:
    pass
layout["Boston_Red_Sox_at_Clevela"] = [0, 2303, 1440, 2596]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/10_icon_Globe_Life_Field.png
try:
    _c10 = get_crop(10, 1440, 293)
    canvas.paste(_c10, (0, 1279), _c10)
except Exception:
    pass
layout["Globe_Life_Field"] = [0, 1279, 1440, 1572]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/11_icon_Globe_Life_Field.png
try:
    _c11 = get_crop(11, 1440, 293)
    canvas.paste(_c11, (0, 1572), _c11)
except Exception:
    pass
layout["Globe_Life_Field"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/12_icon_Boston_Red_Sox_at_Cleveland_Guardians.png
try:
    _c12 = get_crop(12, 1440, 293)
    canvas.paste(_c12, (0, 2596), _c12)
except Exception:
    pass
layout["Boston_Red_Sox_at_Clevela"] = [0, 2596, 1440, 2889]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 85, 102)
    canvas.paste(_c13, (1304, 957), _c13)
except Exception:
    pass
layout["icon_13"] = [1304, 957, 1389, 1059]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/14_icon_Boston_Red_Sox_at_Texas_Rangers.png
try:
    _c14 = get_crop(14, 1440, 293)
    canvas.paste(_c14, (0, 1865), _c14)
except Exception:
    pass
layout["Boston_Red_Sox_at_Texas_R"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/15_icon_Boston_Red_Sox_at_Texas_Rangers.png
try:
    _c15 = get_crop(15, 1440, 293)
    canvas.paste(_c15, (0, 1572), _c15)
except Exception:
    pass
layout["Boston_Red_Sox_at_Texas_R"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 118, 75)
    canvas.paste(_c16, (1212, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1212, 0, 1330, 75]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/17_icon_Globe_Life_Field.png
try:
    _c17 = get_crop(17, 1440, 293)
    canvas.paste(_c17, (0, 1865), _c17)
except Exception:
    pass
layout["Globe_Life_Field"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 63, 75)
    canvas.paste(_c18, (1147, 1), _c18)
except Exception:
    pass
layout["icon_18"] = [1147, 1, 1210, 76]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/19_text_8.35_Wy.png
try:
    _c19 = get_crop(19, 153, 49)
    canvas.paste(_c19, (19, 12), _c19)
except Exception:
    pass
layout["8.35_Wy"] = [19, 12, 172, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/20_text_Boston_Red_Sox.png
try:
    _c20 = get_crop(20, 452, 64)
    canvas.paste(_c20, (57, 859), _c20)
except Exception:
    pass
layout["Boston_Red_Sox"] = [57, 859, 509, 923]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/21_text_Protected_by_our_Buyer_Guarantee.png
try:
    _c21 = get_crop(21, 1440, 126)
    canvas.paste(_c21, (0, 933), _c21)
except Exception:
    pass
layout["Protected_by_our_Buyer_Gu"] = [0, 933, 1440, 1059]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/22_text_Dallas_TX.png
try:
    _c22 = get_crop(22, 271, 69)
    canvas.paste(_c22, (53, 1174), _c22)
except Exception:
    pass
layout["Dallas,_TX"] = [53, 1174, 324, 1243]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/23_text_Rocton_Red_Goy_Ot_Cleveland.png
try:
    _c23 = get_crop(23, 1440, 293)
    canvas.paste(_c23, (0, 2596), _c23)
except Exception:
    pass
layout["Rocton_Red_Goy_Ot_Clevela"] = [0, 2596, 1440, 2889]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_12_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-15/24_text_Guardianc.png
try:
    _c24 = get_crop(24, 240, 29)
    canvas.paste(_c24, (973, 2930), _c24)
except Exception:
    pass
layout["Guardianc"] = [973, 2930, 1213, 2959]
