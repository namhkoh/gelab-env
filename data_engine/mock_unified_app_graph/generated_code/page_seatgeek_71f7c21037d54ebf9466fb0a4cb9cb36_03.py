# page_id: page_seatgeek_71f7c21037d54ebf9466fb0a4cb9cb36_03
# screenshot: 2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6.png
# step_index: 3/4
# task: Open SeatGeek. Search for concerts in "New York City". Filter by "pop" genre. What is the second recommendation?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and UI structural elements for the mobile UI mockup.
# Available variables:
# - canvas: PIL Image (1440x2960 RGB)
# - draw: PIL ImageDraw object
# - font_sm, font_md, font_lg, font_xl

# Colors
BG = (247, 248, 250)            # overall app background (very light gray)
STATUS_BAR = (236, 236, 236)    # status bar strip
HEADER_BG = (255, 255, 255)     # white toolbar/header
DIVIDER = (228, 229, 231)       # subtle divider lines
CARD_BG = (255, 255, 255)       # card white
IMAGE_BG = (240, 241, 243)      # placeholder color for image areas
SHADOW = (222, 223, 224)        # shadow approximation for cards
FILTER_BG = (255, 255, 255)     # filter row background (same as header)

W, H = canvas.size

# Fill overall background
draw.rectangle((0, 0, W, H), fill=BG)

# Status bar area (~ top 50-90px)
status_h = 88
draw.rectangle((0, 0, W, status_h), fill=STATUS_BAR)

# Header / toolbar background (below status bar)
header_top = status_h
header_h = 112
header_bottom = header_top + header_h
draw.rectangle((0, header_top, W, header_bottom), fill=HEADER_BG)
# bottom divider for header
draw.line((24, header_bottom, W-24, header_bottom), fill=DIVIDER, width=1)

# Filter row area (rounded white strip where genre pills sit) - background only
filter_top = header_bottom
filter_h = 140
filter_bottom = filter_top + filter_h
# Put a white background for filter area
draw.rectangle((0, filter_top, W, filter_bottom), fill=FILTER_BG)
# subtle top divider
draw.line((0, filter_top, W, filter_top), fill=DIVIDER, width=1)
# subtle bottom divider
draw.line((0, filter_bottom, W, filter_bottom), fill=DIVIDER, width=1)

# Helper to draw event card background with simple shadow and image area
def draw_event_card(x, y, card_w, card_h, img_ratio=0.45, radius=14):
    # shadow (offset)
    shadow_offset = 8
    draw.rectangle((x+shadow_offset, y+shadow_offset, x+card_w+shadow_offset, y+card_h+shadow_offset),
                   fill=SHADOW)
    # card background (rounded)
    draw.rounded_rectangle((x, y, x+card_w, y+card_h), radius=radius, fill=CARD_BG)
    # image/banner area background (top portion of the card)
    img_h = int(card_h * img_ratio)
    # top-left and top-right corners should appear clipped by card; draw rounded rectangle for image top
    draw.rounded_rectangle((x, y, x+card_w, y+img_h), radius=radius, fill=IMAGE_BG)
    # divider between image and details
    divider_y = y + img_h + 18
    draw.line((x+18, divider_y, x+card_w-18, divider_y), fill=DIVIDER, width=1)
    # subtle bottom divider inside card (above action row)
    action_y = y + card_h - 78
    draw.line((x+18, action_y, x+card_w-18, action_y), fill=DIVIDER, width=1)

# Layout cards vertically with spacing
margin_x = 24
card_w = W - 2 * margin_x

# Card 1 (top event)
card1_y = filter_bottom + 24
card1_h = 540
draw_event_card(margin_x, card1_y, card_w, card1_h)

# Spacer between cards and divider
gap = 36
# thin page separator
sep_y = card1_y + card1_h + gap//2
draw.line((12, sep_y, W-12, sep_y), fill=DIVIDER, width=1)

# Card 2 (middle event)
card2_y = card1_y + card1_h + gap
card2_h = 620
draw_event_card(margin_x, card2_y, card_w, card2_h)

# separator below card 2
sep2_y = card2_y + card2_h + gap//2
draw.line((12, sep2_y, W-12, sep2_y), fill=DIVIDER, width=1)

# Card 3 (lower event preview)
card3_y = card2_y + card2_h + gap
card3_h = 520
draw_event_card(margin_x, card3_y, card_w, card3_h)

# Large bottom padding divider
bottom_div_y = card3_y + card3_h + 32
draw.line((0, bottom_div_y, W, bottom_div_y), fill=DIVIDER, width=8)

# Add thin separators under header elements to visually separate groups
draw.line((24, header_bottom+8, W-24, header_bottom+8), fill=DIVIDER, width=1)
draw.line((24, filter_bottom+6, W-24, filter_bottom+6), fill=DIVIDER, width=1)

# Optional subtle vertical guideline gutters (for visual balance, not icons/text)
gutters_color = (250, 250, 250)
draw.line((margin_x, 0, margin_x, H), fill=gutters_color, width=1)
draw.line((W-margin_x, 0, W-margin_x, H), fill=gutters_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/00_icon_Alternative.png
try:
    _c0 = get_crop(0, 311, 97)
    canvas.paste(_c0, (437, 335), _c0)
except Exception:
    pass
layout["Alternative"] = [437, 335, 748, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/01_icon_Rnb.png
try:
    _c1 = get_crop(1, 171, 97)
    canvas.paste(_c1, (1269, 335), _c1)
except Exception:
    pass
layout["Rnb"] = [1269, 335, 1440, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/02_icon_Pop.png
try:
    _c2 = get_crop(2, 173, 97)
    canvas.paste(_c2, (21, 335), _c2)
except Exception:
    pass
layout["Pop"] = [21, 335, 194, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/03_icon_Folk.png
try:
    _c3 = get_crop(3, 176, 97)
    canvas.paste(_c3, (772, 335), _c3)
except Exception:
    pass
layout["Folk"] = [772, 335, 948, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/04_icon_Hip-Hop.png
try:
    _c4 = get_crop(4, 264, 97)
    canvas.paste(_c4, (972, 335), _c4)
except Exception:
    pass
layout["Hip-Hop"] = [972, 335, 1236, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/05_icon_Rock.png
try:
    _c5 = get_crop(5, 195, 97)
    canvas.paste(_c5, (218, 335), _c5)
except Exception:
    pass
layout["Rock"] = [218, 335, 413, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/06_icon_Track.png
try:
    _c6 = get_crop(6, 267, 185)
    canvas.paste(_c6, (0, 1382), _c6)
except Exception:
    pass
layout["Track"] = [0, 1382, 267, 1567]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/07_icon_Track.png
try:
    _c7 = get_crop(7, 267, 185)
    canvas.paste(_c7, (0, 2517), _c7)
except Exception:
    pass
layout["Track"] = [0, 2517, 267, 2702]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/08_icon_Share.png
try:
    _c8 = get_crop(8, 248, 162)
    canvas.paste(_c8, (267, 1398), _c8)
except Exception:
    pass
layout["Share"] = [267, 1398, 515, 1560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/09_icon_884.png
try:
    _c9 = get_crop(9, 144, 240)
    canvas.paste(_c9, (1260, 72), _c9)
except Exception:
    pass
layout["884"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/10_icon_Morgan_Wallen_Rescheduled_from_6_17_23.png
try:
    _c10 = get_crop(10, 1440, 1135)
    canvas.paste(_c10, (0, 1591), _c10)
except Exception:
    pass
layout["Morgan_Wallen_(Reschedule"] = [0, 1591, 1440, 2726]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/11_icon_Share.png
try:
    _c11 = get_crop(11, 248, 162)
    canvas.paste(_c11, (267, 2533), _c11)
except Exception:
    pass
layout["Share"] = [267, 2533, 515, 2695]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/12_icon_7.03_my.png
try:
    _c12 = get_crop(12, 58, 58)
    canvas.paste(_c12, (113, 4), _c12)
except Exception:
    pass
layout["7.03_my"] = [113, 4, 171, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/13_icon_Concerts.png
try:
    _c13 = get_crop(13, 62, 56)
    canvas.paste(_c13, (242, 7), _c13)
except Exception:
    pass
layout["Concerts"] = [242, 7, 304, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/14_icon_7.03_my.png
try:
    _c14 = get_crop(14, 144, 240)
    canvas.paste(_c14, (0, 72), _c14)
except Exception:
    pass
layout["7.03_my"] = [0, 72, 144, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/15_icon_Concerts.png
try:
    _c15 = get_crop(15, 54, 57)
    canvas.paste(_c15, (314, 7), _c15)
except Exception:
    pass
layout["Concerts"] = [314, 7, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/16_icon_7.03_my.png
try:
    _c16 = get_crop(16, 49, 56)
    canvas.paste(_c16, (184, 6), _c16)
except Exception:
    pass
layout["7.03_my"] = [184, 6, 233, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 47, 65)
    canvas.paste(_c17, (1154, 1), _c17)
except Exception:
    pass
layout["icon_17"] = [1154, 1, 1201, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 48, 56)
    canvas.paste(_c18, (1321, 5), _c18)
except Exception:
    pass
layout["icon_18"] = [1321, 5, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/19_icon_884.png
try:
    _c19 = get_crop(19, 89, 61)
    canvas.paste(_c19, (1216, 2), _c19)
except Exception:
    pass
layout["884"] = [1216, 2, 1305, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/20_icon_S218.png
try:
    _c20 = get_crop(20, 183, 68)
    canvas.paste(_c20, (38, 2182), _c20)
except Exception:
    pass
layout["S218+"] = [38, 2182, 221, 2250]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 44, 60)
    canvas.paste(_c21, (385, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [385, 3, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/22_icon_Share.png
try:
    _c22 = get_crop(22, 532, 176)
    canvas.paste(_c22, (448, 2783), _c22)
except Exception:
    pass
layout["Share"] = [448, 2783, 980, 2959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/23_icon_New_York_NY.png
try:
    _c23 = get_crop(23, 1440, 1135)
    canvas.paste(_c23, (0, 456), _c23)
except Exception:
    pass
layout["New_York,_NY"] = [0, 456, 1440, 1591]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/24_text_Concerts.png
try:
    _c24 = get_crop(24, 267, 63)
    canvas.paste(_c24, (186, 133), _c24)
except Exception:
    pass
layout["Concerts"] = [186, 133, 453, 196]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/25_text_New_York_NY.png
try:
    _c25 = get_crop(25, 195, 97)
    canvas.paste(_c25, (218, 335), _c25)
except Exception:
    pass
layout["New_York,_NY"] = [218, 335, 413, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/26_text_date.png
try:
    _c26 = get_crop(26, 117, 52)
    canvas.paste(_c26, (606, 208), _c26)
except Exception:
    pass
layout["date"] = [606, 208, 723, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/27_text_Morgan_Wallen_Rescheduled_from_6_17_23.png
try:
    _c27 = get_crop(27, 1440, 1135)
    canvas.paste(_c27, (0, 1591), _c27)
except Exception:
    pass
layout["Morgan_Wallen_(Reschedule"] = [0, 1591, 1440, 2726]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/28_text_Sat.png
try:
    _c28 = get_crop(28, 102, 56)
    canvas.paste(_c28, (40, 2411), _c28)
except Exception:
    pass
layout["Sat,"] = [40, 2411, 142, 2467]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/29_text_11_4_PM.png
try:
    _c29 = get_crop(29, 170, 50)
    canvas.paste(_c29, (241, 2414), _c29)
except Exception:
    pass
layout["11,4_PM"] = [241, 2414, 411, 2464]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/30_text_Philadelphia_PA.png
try:
    _c30 = get_crop(30, 346, 66)
    canvas.paste(_c30, (433, 2406), _c30)
except Exception:
    pass
layout["Philadelphia,_PA"] = [433, 2406, 779, 2472]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_03_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-6/31_text_Citizens_Bank_Park.png
try:
    _c31 = get_crop(31, 405, 50)
    canvas.paste(_c31, (803, 2412), _c31)
except Exception:
    pass
layout["Citizens_Bank_Park"] = [803, 2412, 1208, 2462]
