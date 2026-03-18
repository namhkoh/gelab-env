# page_id: page_seatgeek_275896c29e4341b1a61db6ff596ab44c_05
# screenshot: 2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8.png
# step_index: 5/9
# task: Open SeatGeek. Look up "Seattle Mariners" tickets. Select the next upcoming event in Los Angeles. Set quantity to 2 and select the best value tickets. Which section are tickets in?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure elements on provided canvas and draw objects.
# Available: canvas (1440x2960 RGB), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Full background fill (page background - subtle off-white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFBFB")

# STATUS BAR (top area ~88px tall) - dark/navy bar behind status icons
status_h = 88
draw.rectangle([(0, 0), (1440, status_h)], fill="#0B3B5A")

# HERO BANNER AREA (image area placeholder background)
hero_top = status_h
hero_bottom = 480
draw.rectangle([(0, hero_top), (1440, hero_bottom)], fill="#12334A")

# Thin dark divider under hero banner
draw.line([(0, hero_bottom), (1440, hero_bottom)], fill="#0E2B40", width=6)

# Subtle darker band near the bottom of the banner to simulate image edge
draw.rectangle([(0, hero_bottom - 10), (1440, hero_bottom)], fill="#0B2738")

# MAIN CARD (rounded white card overlapping the hero banner)
card_left = 40
card_top = hero_bottom - 60   # overlap hero
card_right = 1400
card_bottom = 760
draw.rounded_rectangle(
    [(card_left, card_top), (card_right, card_bottom)],
    radius=22,
    fill="#FFFFFF",
    outline="#E7E7E7",
)

# Thin separator under the card to accentuate separation
draw.line([(card_left + 8, card_bottom), (card_right - 8, card_bottom)], fill="#ECECEC", width=1)

# Content list panel (holds event rows) - large white rounded panel
list_left = 32
list_top = 840
list_right = 1408
list_bottom = 2520
draw.rounded_rectangle(
    [(list_left, list_top), (list_right, list_bottom)],
    radius=12,
    fill="#FFFFFF",
    outline="#EDEDED",
    width=1
)

# Subtle shadow-like top edge for the list panel
draw.line([(list_left + 6, list_top), (list_right - 6, list_top)], fill="#F5F5F5", width=3)

# Section separators between groups (light lines)
separator_x1 = list_left + 40
separator_x2 = list_right - 40
separators = [1120, 1400, 1680, 1960, 2240]
for y in separators:
    if list_top + 10 < y < list_bottom - 10:
        draw.line([(separator_x1, y), (separator_x2, y)], fill="#F2F2F2", width=1)

# Light left inset vertical guide (visual gutter for date pills area)
gutter_x = list_left + 110
draw.line([(gutter_x, list_top + 20), (gutter_x, list_bottom - 20)], fill="#FAFAFA", width=1)

# Section header areas (subtle label backgrounds to separate major groups)
# "Los Angeles, CA" header bar (background hint only)
hdr1_top = 900
hdr1_bottom = hdr1_top + 84
draw.rectangle([(list_left + 12, hdr1_top), (list_right - 12, hdr1_bottom)], fill="#FFFFFF")
draw.line([(list_left + 12, hdr1_bottom), (list_right - 12, hdr1_bottom)], fill="#EFEFEF", width=1)

# "All Games" header bar hint
hdr2_top = 2360 - 140  # place above all-games list
hdr2_bottom = hdr2_top + 72
if hdr2_top > list_top and hdr2_bottom < list_bottom:
    draw.rectangle([(list_left + 12, hdr2_top), (list_right - 12, hdr2_bottom)], fill="#FFFFFF")
    draw.line([(list_left + 12, hdr2_bottom), (list_right - 12, hdr2_bottom)], fill="#EFEFEF", width=1)

# Bottom footer area (background) - leave room for pasted bottom icons
footer_top = 2596
draw.rectangle([(0, footer_top), (1440, 2960)], fill="#FFFFFF")
draw.line([(0, footer_top), (1440, footer_top)], fill="#E9E9E9", width=1)

# Decorative accent bar matching app brand color under the hero/card juncture
accent_y = card_top + 12
draw.rectangle([(card_left + 8, accent_y), (card_left + 8 + 220, accent_y + 8)], fill="#12324A")

# Small divider under hero image area (across full width) to separate image from content
draw.line([(20, hero_bottom + 6), (1420, hero_bottom + 6)], fill="#F0F0F0", width=1)

# Slight inner shadow along the right edge of list panel for depth
for i in range(6):
    alpha = 220 - i * 30
    y1 = list_top + 8 + i*2
    y2 = list_bottom - 8 - i*2
    draw.line([(list_right - 2 - i, y1), (list_right - 2 - i, y2)], fill="#F7F7F7", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/00_icon_13.png
try:
    _c0 = get_crop(0, 1440, 293)
    canvas.paste(_c0, (0, 1865), _c0)
except Exception:
    pass
layout["13"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/01_icon_14.png
try:
    _c1 = get_crop(1, 1440, 293)
    canvas.paste(_c1, (0, 2158), _c1)
except Exception:
    pass
layout["14"] = [0, 2158, 1440, 2451]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/02_icon_11.png
try:
    _c2 = get_crop(2, 1440, 293)
    canvas.paste(_c2, (0, 1279), _c2)
except Exception:
    pass
layout["11"] = [0, 1279, 1440, 1572]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/03_icon_23.png
try:
    _c3 = get_crop(3, 1440, 293)
    canvas.paste(_c3, (0, 2596), _c3)
except Exception:
    pass
layout["23"] = [0, 2596, 1440, 2889]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/04_icon_12.png
try:
    _c4 = get_crop(4, 1440, 293)
    canvas.paste(_c4, (0, 1572), _c4)
except Exception:
    pass
layout["12"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/05_icon_Seattle_Mariners.png
try:
    _c5 = get_crop(5, 204, 201)
    canvas.paste(_c5, (51, 603), _c5)
except Exception:
    pass
layout["Seattle_Mariners"] = [51, 603, 255, 804]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/06_icon_Iy.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (36, 84), _c6)
except Exception:
    pass
layout["Iy"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/07_icon_Share_this_performer.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1260, 84), _c7)
except Exception:
    pass
layout["Share_this_performer"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/08_icon_Track_this_performer.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1104, 84), _c8)
except Exception:
    pass
layout["Track_this_performer"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/09_icon_Angel_Stadium_of_Anaheim.png
try:
    _c9 = get_crop(9, 1440, 293)
    canvas.paste(_c9, (0, 1279), _c9)
except Exception:
    pass
layout["Angel_Stadium_of_Anaheim"] = [0, 1279, 1440, 1572]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/10_icon_Angel_Stadium_of_Anaheim.png
try:
    _c10 = get_crop(10, 1440, 293)
    canvas.paste(_c10, (0, 1572), _c10)
except Exception:
    pass
layout["Angel_Stadium_of_Anaheim"] = [0, 1572, 1440, 1865]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 60, 76)
    canvas.paste(_c11, (1150, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [1150, 2, 1210, 78]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 80, 95)
    canvas.paste(_c12, (1307, 960), _c12)
except Exception:
    pass
layout["icon_12"] = [1307, 960, 1387, 1055]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/13_icon_Angel_Stadium_of_Anaheim.png
try:
    _c13 = get_crop(13, 1440, 293)
    canvas.paste(_c13, (0, 1865), _c13)
except Exception:
    pass
layout["Angel_Stadium_of_Anaheim"] = [0, 1865, 1440, 2158]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 52, 61)
    canvas.paste(_c14, (1320, 4), _c14)
except Exception:
    pass
layout["icon_14"] = [1320, 4, 1372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/15_icon_7.48.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (36, 84), _c15)
except Exception:
    pass
layout["7.48"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 97, 63)
    canvas.paste(_c16, (1217, 4), _c16)
except Exception:
    pass
layout["icon_16"] = [1217, 4, 1314, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/17_icon_Arlington_TX.png
try:
    _c17 = get_crop(17, 1440, 293)
    canvas.paste(_c17, (0, 2596), _c17)
except Exception:
    pass
layout["Arlington,_TX"] = [0, 2596, 1440, 2889]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/18_text_Seattle_Mariners.png
try:
    _c18 = get_crop(18, 486, 64)
    canvas.paste(_c18, (55, 859), _c18)
except Exception:
    pass
layout["Seattle_Mariners"] = [55, 859, 541, 923]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/19_text_Protected_by_our_Buyer_Guarantee.png
try:
    _c19 = get_crop(19, 1440, 126)
    canvas.paste(_c19, (0, 933), _c19)
except Exception:
    pass
layout["Protected_by_our_Buyer_Gu"] = [0, 933, 1440, 1059]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/20_text_Los_Angeles_CA.png
try:
    _c20 = get_crop(20, 421, 75)
    canvas.paste(_c20, (54, 1175), _c20)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [54, 1175, 475, 1250]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/21_text_Seattle_Mariners_at_Los_Angeles_Angels.png
try:
    _c21 = get_crop(21, 1440, 293)
    canvas.paste(_c21, (0, 2158), _c21)
except Exception:
    pass
layout["Seattle_Mariners_at_Los_A"] = [0, 2158, 1440, 2451]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/22_text_1.07_PM.png
try:
    _c22 = get_crop(22, 168, 49)
    canvas.paste(_c22, (312, 2276), _c22)
except Exception:
    pass
layout["1.07_PM"] = [312, 2276, 480, 2325]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/23_text_Angel_Stadium_of_Anaheim.png
try:
    _c23 = get_crop(23, 1440, 293)
    canvas.paste(_c23, (0, 2158), _c23)
except Exception:
    pass
layout["Angel_Stadium_of_Anaheim"] = [0, 2158, 1440, 2451]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/24_text_From_S11.png
try:
    _c24 = get_crop(24, 186, 54)
    canvas.paste(_c24, (315, 2361), _c24)
except Exception:
    pass
layout["From_S11"] = [315, 2361, 501, 2415]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_05_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-8/25_text_AIl_Games.png
try:
    _c25 = get_crop(25, 265, 55)
    canvas.paste(_c25, (60, 2495), _c25)
except Exception:
    pass
layout["AIl_Games"] = [60, 2495, 325, 2550]
