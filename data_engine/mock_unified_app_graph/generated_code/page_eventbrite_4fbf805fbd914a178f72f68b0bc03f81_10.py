# page_id: page_eventbrite_4fbf805fbd914a178f72f68b0bc03f81_10
# screenshot: 2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12.png
# step_index: 10/10
# task: Open Eventbrite. Explore "Education" events. Apply filters for events happening tomorrow. From the list, select the third event and check out its description.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the page
# Uses provided variables: canvas (PIL Image), draw (PIL ImageDraw),
# and fonts: font_sm, font_md, font_lg, font_xl

# Colors
bg_color = "#FBFBFD"            # overall page background (very light)
status_bar_color = "#E9E9EA"    # top status bar
header_bg = "#FFFFFF"           # header/toolbar background
divider_color = "#EFEFF2"       # subtle dividers
card_bg = "#FFFFFF"             # card interiors
card_border = "#E8E8EE"         # card border
selected_blue = "#2F47FF"       # accent blue for selected card border
muted_gray = "#F6F6F8"          # subtle background for shadows/pills
reservation_border = "#3750FF"  # blue border for reservation card
reservation_shadow = "#E9E9ED"
primary_action = "#C94A1A"      # reserve button orange

W, H = canvas.size

# Fill overall background
draw.rectangle((0, 0, W, H), fill=bg_color)

# Status bar (top ~72px)
status_h = 72
draw.rectangle((0, 0, W, status_h), fill=status_bar_color)

# Header / toolbar area under status bar
header_top = status_h
header_bottom = 160
draw.rectangle((0, header_top, W, header_bottom), fill=header_bg)
# subtle bottom divider line
draw.line((24, header_bottom, W - 24, header_bottom), fill=divider_color, width=2)

# Row of date/time selection cards
# Use detected icon positions to place rounded card backgrounds behind them.
# Cards correspond to detected icon positions:
cards = [
    {"pos": (24, 527), "size": (450, 516), "selected": True},
    {"pos": (474, 527), "size": (450, 516), "selected": False},
    {"pos": (924, 527), "size": (450, 516), "selected": False},
]

for i, card in enumerate(cards):
    x, y = card["pos"]
    w_card, h_card = card["size"]
    # expand the card rect a bit to include label area above and padding
    left = x - 12
    right = x + w_card + 12
    top = max( header_bottom + 20, y - 190 )   # ensure below header
    bottom = y + h_card - 40
    radius = 28

    # card shadow/background subtle
    shadow_rect = (left + 6, top + 10, right + 6, bottom + 10)
    draw.rounded_rectangle(shadow_rect, radius=radius, fill=muted_gray)

    # card interior
    draw.rounded_rectangle((left, top, right, bottom), radius=radius, fill=card_bg, outline=card_border, width=3)

    # if selected, draw a thicker blue outline
    if card.get("selected"):
        draw.rounded_rectangle((left+2, top+2, right-2, bottom-2), radius=radius-2, outline=selected_blue, width=6)

# Thin separator under date cards
sep_y = cards[0]["pos"][1] + cards[0]["size"][1] - 20
draw.line((24, sep_y + 36, W - 24, sep_y + 36), fill=divider_color, width=2)

# "About this event" section background band (subtle)
about_top = sep_y + 80
about_bottom = about_top + 720
# keep background same as page but add a faint top divider and bottom divider
draw.line((24, about_top, W - 24, about_top), fill=divider_color, width=1)
draw.line((24, about_bottom, W - 24, about_bottom), fill=divider_color, width=1)

# Decorative small badge area (background for category pill) - do not draw text
pill_left = 36
pill_top = about_top + 28
pill_right = pill_left + 420
pill_bottom = pill_top + 64
draw.rounded_rectangle((pill_left, pill_top, pill_right, pill_bottom), radius=32, fill="#F2F4F8")

# Subtle horizontal rule between description blocks
desc_line_y = about_top + 220
draw.line((36, desc_line_y, W - 36, desc_line_y), fill=divider_color, width=1)

# A faint left decorative bar to visually group the descriptive text area
decor_left = 36
decor_top = desc_line_y + 24
decor_bottom = about_bottom - 80
draw.rectangle((decor_left, decor_top, decor_left + 6, decor_bottom), fill="#F0EAF6")  # very subtle tint

# Reservation card area (rounded white card with blue border)
# Use detected positions for increase/decrease icons to estimate placement
res_card_top = 2320
res_card_left = 48
res_card_right = W - 48
res_card_bottom = 2560
res_radius = 22

# shadow for reservation card
draw.rounded_rectangle((res_card_left + 6, res_card_top + 8, res_card_right + 6, res_card_bottom + 8),
                       radius=res_radius, fill=reservation_shadow)

# main reservation card with blue border
draw.rounded_rectangle((res_card_left, res_card_top, res_card_right, res_card_bottom),
                       radius=res_radius, fill=card_bg, outline=reservation_border, width=6)

# small inner dividing line inside reservation card (visual separation)
inner_div_y = res_card_top + 84
draw.line((res_card_left + 36, inner_div_y, res_card_right - 36, inner_div_y), fill=divider_color, width=1)

# Thin separator above the bottom action button
action_sep_y = 2736
draw.line((36, action_sep_y, W - 36, action_sep_y), fill=divider_color, width=1)

# Primary action "Reserve a spot" button background (do not draw the label text)
button_left = 72
button_top = 2756
button_right = W - 72
button_bottom = button_top + 132
button_radius = 12
# subtle shadow behind button
draw.rounded_rectangle((button_left, button_top + 6, button_right, button_bottom + 6),
                       radius=button_radius+2, fill="#EDE1DB")
# button fill
draw.rounded_rectangle((button_left, button_top, button_right, button_bottom),
                       radius=button_radius, fill=primary_action)

# Final subtle bottom safe area fill (slightly darker to ground the bottom)
draw.rectangle((0, H - 24, W, H), fill="#F5F5F6")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/00_icon_24.png
try:
    _c0 = get_crop(0, 450, 516)
    canvas.paste(_c0, (24, 527), _c0)
except Exception:
    pass
layout["24"] = [24, 527, 474, 1043]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/01_icon_25.png
try:
    _c1 = get_crop(1, 450, 516)
    canvas.paste(_c1, (474, 527), _c1)
except Exception:
    pass
layout["25"] = [474, 527, 924, 1043]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/02_icon_More.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1116, 108), _c2)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/03_icon_Share.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1260, 108), _c3)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/04_icon_Decrease.png
try:
    _c4 = get_crop(4, 99, 96)
    canvas.paste(_c4, (996, 2441), _c4)
except Exception:
    pass
layout["Decrease"] = [996, 2441, 1095, 2537]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/05_icon_Increase.png
try:
    _c5 = get_crop(5, 96, 96)
    canvas.paste(_c5, (1224, 2441), _c5)
except Exception:
    pass
layout["Increase"] = [1224, 2441, 1320, 2537]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 93, 102)
    canvas.paste(_c6, (1107, 2439), _c6)
except Exception:
    pass
layout["icon_6"] = [1107, 2439, 1200, 2541]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/07_icon_Reserve_a_spot.png
try:
    _c7 = get_crop(7, 1296, 132)
    canvas.paste(_c7, (72, 2756), _c7)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/08_icon_4.56.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (36, 108), _c8)
except Exception:
    pass
layout["4.56"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/09_icon_27.png
try:
    _c9 = get_crop(9, 450, 516)
    canvas.paste(_c9, (924, 527), _c9)
except Exception:
    pass
layout["27"] = [924, 527, 1374, 1043]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 54, 55)
    canvas.paste(_c10, (314, 6), _c10)
except Exception:
    pass
layout["icon_10"] = [314, 6, 368, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 71, 80)
    canvas.paste(_c11, (1139, 1683), _c11)
except Exception:
    pass
layout["icon_11"] = [1139, 1683, 1210, 1763]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 57, 56)
    canvas.paste(_c12, (182, 5), _c12)
except Exception:
    pass
layout["icon_12"] = [182, 5, 239, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 46, 52)
    canvas.paste(_c13, (251, 8), _c13)
except Exception:
    pass
layout["icon_13"] = [251, 8, 297, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 98, 58)
    canvas.paste(_c14, (1215, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [1215, 2, 1313, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/15_icon_Business_Professional.png
try:
    _c15 = get_crop(15, 772, 99)
    canvas.paste(_c15, (35, 1290), _c15)
except Exception:
    pass
layout["Business_&_Professional"] = [35, 1290, 807, 1389]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 44, 56)
    canvas.paste(_c16, (1326, 4), _c16)
except Exception:
    pass
layout["icon_16"] = [1326, 4, 1370, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/17_icon_4.56.png
try:
    _c17 = get_crop(17, 57, 59)
    canvas.paste(_c17, (116, 3), _c17)
except Exception:
    pass
layout["4.56"] = [116, 3, 173, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/18_icon_27.png
try:
    _c18 = get_crop(18, 450, 516)
    canvas.paste(_c18, (924, 527), _c18)
except Exception:
    pass
layout["27"] = [924, 527, 1374, 1043]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/19_icon_By_Invitation_Only.png
try:
    _c19 = get_crop(19, 99, 96)
    canvas.paste(_c19, (996, 2441), _c19)
except Exception:
    pass
layout["By_Invitation_Only"] = [996, 2441, 1095, 2537]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/20_icon_Free.png
try:
    _c20 = get_crop(20, 75, 72)
    canvas.paste(_c20, (249, 2585), _c20)
except Exception:
    pass
layout["Free"] = [249, 2585, 324, 2657]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/21_icon_satisfy_your_thirst_for_learning_Let_s_e.png
try:
    _c21 = get_crop(21, 99, 96)
    canvas.paste(_c21, (996, 2441), _c21)
except Exception:
    pass
layout["satisfy_your_thirst_for_l"] = [996, 2441, 1095, 2537]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/22_icon_Ignite_your_financial_future_with_knowle.png
try:
    _c22 = get_crop(22, 450, 516)
    canvas.paste(_c22, (474, 527), _c22)
except Exception:
    pass
layout["Ignite_your_financial_fut"] = [474, 527, 924, 1043]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 71, 74)
    canvas.paste(_c23, (40, 1556), _c23)
except Exception:
    pass
layout["icon_23"] = [40, 1556, 111, 1630]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 48, 58)
    canvas.paste(_c24, (383, 4), _c24)
except Exception:
    pass
layout["icon_24"] = [383, 4, 431, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/25_text_4.56.png
try:
    _c25 = get_crop(25, 92, 43)
    canvas.paste(_c25, (22, 17), _c25)
except Exception:
    pass
layout["4.56"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/26_text_The_Path_to_Wealth_T_..png
try:
    _c26 = get_crop(26, 144, 144)
    canvas.paste(_c26, (36, 108), _c26)
except Exception:
    pass
layout["The_Path_to_Wealth_T_."] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/27_text_Select_date_and_time.png
try:
    _c27 = get_crop(27, 144, 144)
    canvas.paste(_c27, (36, 108), _c27)
except Exception:
    pass
layout["Select_date_and_time"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_10_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-12/28_text_About_this_event.png
try:
    _c28 = get_crop(28, 453, 65)
    canvas.paste(_c28, (44, 1200), _c28)
except Exception:
    pass
layout["About_this_event"] = [44, 1200, 497, 1265]
