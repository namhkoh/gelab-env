# page_id: page_seatgeek_6d3c2be0a0b34daf904d1c72c351bd6e_05
# screenshot: 2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8.png
# step_index: 5/9
# task: Open SeatGeek. Look up "Phoenix Suns" tickets for next upcoming event. Which section are tickets in?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background
draw.rectangle([(0, 0), (1440, 2960)], fill=(250, 250, 250))

# Status bar (dark strip at very top)
STATUS_H = 96
draw.rectangle([(0, 0), (1440, STATUS_H)], fill=(22, 22, 22))

# Large header image area (placeholder dark image background)
HEADER_TOP = STATUS_H
HEADER_BOTTOM = 640
# base dark area
draw.rectangle([(0, HEADER_TOP), (1440, HEADER_BOTTOM)], fill=(36, 36, 36))
# subtle vertical gradient overlay to emulate photo depth
for i in range(0, HEADER_BOTTOM - HEADER_TOP):
    alpha = int(12 * (i / max(1, HEADER_BOTTOM - HEADER_TOP)))
    y = HEADER_TOP + i
    draw.line([(0, y), (1440, y)], fill=(36 - alpha, 36 - alpha, 36 - alpha))

# Divider line under header
draw.line([(40, HEADER_BOTTOM), (1400, HEADER_BOTTOM)], fill=(230, 230, 230), width=1)

# Main white content card with rounded top corners overlapping the header
CARD_TOP = HEADER_BOTTOM - 40  # overlap to create the curved look
CARD_LEFT = 24
CARD_RIGHT = 1440 - 24
CARD_BOTTOM = 2960 - 40
draw.rounded_rectangle(
    [(CARD_LEFT, CARD_TOP), (CARD_RIGHT, CARD_BOTTOM)],
    radius=28,
    fill=(255, 255, 255),
    outline=None
)

# Thin soft shadow line under the header/card junction
draw.line([(CARD_LEFT + 8, CARD_TOP + 1), (CARD_RIGHT - 8, CARD_TOP + 1)], fill=(235, 235, 235), width=2)

# "Protected by our Buyer Guarantee" row divider area (subtle separator)
SEP_Y_1 = CARD_TOP + 140
draw.line([(40, SEP_Y_1), (1400, SEP_Y_1)], fill=(245, 245, 245), width=1)

# Card area for the "No Games near ..." section (rounded inner card)
NO_GAMES_TOP = CARD_TOP + 40
NO_GAMES_BOTTOM = NO_GAMES_TOP + 240
draw.rounded_rectangle(
    [(48, NO_GAMES_TOP), (1392, NO_GAMES_BOTTOM)],
    radius=14,
    fill=(255, 255, 255),
    outline=(235, 235, 235)
)
# light inner divider inside this card
draw.line([(64, NO_GAMES_BOTTOM), (1376, NO_GAMES_BOTTOM)], fill=(245, 245, 245), width=1)

# Track button background placeholder (just the pill background, not the text)
TRACK_BTN_LEFT = 64
TRACK_BTN_TOP = NO_GAMES_TOP + 120
TRACK_BTN_RIGHT = TRACK_BTN_LEFT + 260
TRACK_BTN_BOTTOM = TRACK_BTN_TOP + 78
draw.rounded_rectangle(
    [(TRACK_BTN_LEFT, TRACK_BTN_TOP), (TRACK_BTN_RIGHT, TRACK_BTN_BOTTOM)],
    radius=16,
    fill=(255, 255, 255),
    outline=(220, 220, 220)
)

# "All Games" section header spacing divider
ALL_GAMES_Y = NO_GAMES_BOTTOM + 40
draw.line([(40, ALL_GAMES_Y), (1400, ALL_GAMES_Y)], fill=(235, 235, 235), width=1)

# List background area for game items
LIST_TOP = ALL_GAMES_Y + 40
LIST_LEFT = 48
LIST_RIGHT = 1392
LIST_BOTTOM = CARD_BOTTOM - 60
# draw subtle background block (keeps items readable)
draw.rectangle([(LIST_LEFT, LIST_TOP), (LIST_RIGHT, LIST_BOTTOM)], fill=(255, 255, 255))

# Repeating separators for list items (visual structure only)
ITEM_HEIGHT = 280  # approximate height per list entry grouping
y = LIST_TOP + 8
while y < LIST_BOTTOM - 8:
    # subtle separator line between items
    draw.line([(LIST_LEFT + 8, y + ITEM_HEIGHT - 16), (LIST_RIGHT - 8, y + ITEM_HEIGHT - 16)], fill=(245, 245, 245), width=1)
    # left date pill background (rounded) - empty background only
    pill_x0 = LIST_LEFT + 8
    pill_y0 = y + 16
    pill_x1 = pill_x0 + 160
    pill_y1 = pill_y0 + 200
    draw.rounded_rectangle([(pill_x0, pill_y0), (pill_x1, pill_y1)], radius=18, fill=(250, 250, 250), outline=(235, 235, 235))
    # subtle right-hand chevron area background to indicate tappable row (no icon text)
    chevron_box = [(LIST_RIGHT - 80, y + 60), (LIST_RIGHT - 8, y + 200)]
    draw.rectangle(chevron_box, fill=(255, 255, 255))
    draw.line([(LIST_LEFT + 8, y + ITEM_HEIGHT), (LIST_RIGHT - 8, y + ITEM_HEIGHT)], fill=(245, 245, 245), width=1)
    y += ITEM_HEIGHT

# Bottom safe area/footer light divider
FOOTER_Y = CARD_BOTTOM - 40
draw.line([(40, FOOTER_Y), (1400, FOOTER_Y)], fill=(245, 245, 245), width=1)

# final subtle vignette along sides to add depth (very light)
for i in range(1, 12):
    shade = 255 - i  # very small change
    # left small fade
    draw.line([(0 + i, HEADER_BOTTOM), (0 + i, CARD_BOTTOM)], fill=(shade, shade, shade))
    # right small fade
    draw.line([(1440 - i, HEADER_BOTTOM), (1440 - i, CARD_BOTTOM)], fill=(shade, shade, shade))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_05_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8/00_icon_Track_Now.png
try:
    _c0 = get_crop(0, 337, 153)
    canvas.paste(_c0, (60, 1376), _c0)
except Exception:
    pass
layout["Track_Now"] = [60, 1376, 397, 1529]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_05_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8/01_icon_Minneapolis_MN.png
try:
    _c1 = get_crop(1, 1440, 367)
    canvas.paste(_c1, (0, 1785), _c1)
except Exception:
    pass
layout["Minneapolis,_MN"] = [0, 1785, 1440, 2152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_05_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8/02_icon_Timberwolves_at_Phoenix_Suns_Game_3.png
try:
    _c2 = get_crop(2, 1440, 367)
    canvas.paste(_c2, (0, 2152), _c2)
except Exception:
    pass
layout["Timberwolves_at_Phoenix_S"] = [0, 2152, 1440, 2519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_05_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8/03_icon_28.png
try:
    _c3 = get_crop(3, 1440, 367)
    canvas.paste(_c3, (0, 2519), _c3)
except Exception:
    pass
layout["28"] = [0, 2519, 1440, 2886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_05_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8/04_icon_Share_this_performer.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1260, 84), _c4)
except Exception:
    pass
layout["Share_this_performer"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_05_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8/05_icon_6af.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1104, 84), _c5)
except Exception:
    pass
layout["6af"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_05_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8/06_icon_26.png
try:
    _c6 = get_crop(6, 1440, 367)
    canvas.paste(_c6, (0, 2152), _c6)
except Exception:
    pass
layout["26"] = [0, 2152, 1440, 2519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_05_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8/07_icon_23.png
try:
    _c7 = get_crop(7, 1440, 367)
    canvas.paste(_c7, (0, 1785), _c7)
except Exception:
    pass
layout["23"] = [0, 1785, 1440, 2152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_05_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8/08_icon_7.06_Wy.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (36, 84), _c8)
except Exception:
    pass
layout["7.06_Wy"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_05_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8/09_icon_Timberwolves_at_Phoenix_Suns_Game_4.png
try:
    _c9 = get_crop(9, 1440, 367)
    canvas.paste(_c9, (0, 2519), _c9)
except Exception:
    pass
layout["Timberwolves_at_Phoenix_S"] = [0, 2519, 1440, 2886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_05_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 86, 108)
    canvas.paste(_c10, (1305, 950), _c10)
except Exception:
    pass
layout["icon_10"] = [1305, 950, 1391, 1058]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_05_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 49, 66)
    canvas.paste(_c11, (1154, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [1154, 2, 1203, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_05_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8/12_icon_Sun.png
try:
    _c12 = get_crop(12, 211, 55)
    canvas.paste(_c12, (56, 2905), _c12)
except Exception:
    pass
layout["Sun"] = [56, 2905, 267, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_05_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 56, 64)
    canvas.paste(_c13, (1310, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [1310, 3, 1366, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_05_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8/14_text_7.06_Wy.png
try:
    _c14 = get_crop(14, 156, 52)
    canvas.paste(_c14, (19, 12), _c14)
except Exception:
    pass
layout["7.06_Wy"] = [19, 12, 175, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_05_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8/15_text_CDICO.png
try:
    _c15 = get_crop(15, 64, 18)
    canvas.paste(_c15, (688, 70), _c15)
except Exception:
    pass
layout["CDICO"] = [688, 70, 752, 88]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_05_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8/16_text_Phoenix_Suns.png
try:
    _c16 = get_crop(16, 389, 64)
    canvas.paste(_c16, (57, 859), _c16)
except Exception:
    pass
layout["Phoenix_Suns"] = [57, 859, 446, 923]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_05_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8/17_text_Protected_by_our_Buyer_Guarantee.png
try:
    _c17 = get_crop(17, 1440, 126)
    canvas.paste(_c17, (0, 933), _c17)
except Exception:
    pass
layout["Protected_by_our_Buyer_Gu"] = [0, 933, 1440, 1059]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_05_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8/18_text_No_Games_near_New_York_NY.png
try:
    _c18 = get_crop(18, 337, 153)
    canvas.paste(_c18, (60, 1376), _c18)
except Exception:
    pass
layout["No_Games_near_New_York,_N"] = [60, 1376, 397, 1529]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_05_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8/19_text_Track_Phoenix_Suns_for_event_updates.png
try:
    _c19 = get_crop(19, 337, 153)
    canvas.paste(_c19, (60, 1376), _c19)
except Exception:
    pass
layout["Track_Phoenix_Suns_for_ev"] = [60, 1376, 397, 1529]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_05_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8/20_text_AIl_Games.png
try:
    _c20 = get_crop(20, 268, 60)
    canvas.paste(_c20, (59, 1682), _c20)
except Exception:
    pass
layout["AIl_Games"] = [59, 1682, 327, 1742]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_05_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8/21_text_Western_Conference_First_Round_Phoenix_S.png
try:
    _c21 = get_crop(21, 1440, 367)
    canvas.paste(_c21, (0, 2519), _c21)
except Exception:
    pass
layout["Western_Conference_First_"] = [0, 2519, 1440, 2886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_05_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-8/22_text_6af.png
try:
    _c22 = get_crop(22, 83, 60)
    canvas.paste(_c22, (878, 175), _c22)
except Exception:
    pass
layout["6af"] = [878, 175, 961, 235]
