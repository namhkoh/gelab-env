# page_id: page_eventbrite_66847fb559f849b19cea93b83307fae7_03
# screenshot: 2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5.png
# step_index: 3/4
# task: Open Eventbrite. Open favorites and select the second event. Process to checkout and see what payment options it offers.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI elements for Event page (PIL drawing)
# Assumes variables provided: canvas (PIL.Image 1440x2960, white), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Colors
bg_white = (255, 255, 255)
status_bar_gray = (200, 203, 206)        # light gray status bar
hero_top = (30, 30, 30)                  # dark hero area (image placeholder background)
hero_bottom = (60, 60, 60)               # darker bottom of hero
divider_gray = (235, 235, 238)           # subtle divider
card_fill = (249, 247, 250)              # very light card bg (slight purple tint)
card_border = (225, 223, 230)            # card border
muted_purple = (46, 12, 56)              # deep purple accent for subtle strokes
accent_blue = (46, 95, 255)              # bright blue for selection outlines
accent_orange = (200, 62, 15)            # checkout button color
shadow_color = (0, 0, 0, 30)

# Clear background (canvas should already be white)
draw.rectangle([(0,0),(W,H)], fill=bg_white)

# STATUS BAR (top ~56px)
status_h = 56
draw.rectangle([(0,0),(W,status_h)], fill=status_bar_gray)

# Top subtle bottom border of status bar
draw.line([(0,status_h-1),(W,status_h-1)], fill=(190,190,190), width=1)

# HERO IMAGE AREA (placeholder dark gradient)
hero_top_y = status_h
hero_bottom_y = 460
hero_h = hero_bottom_y - hero_top_y

# Vertical gradient for hero area (dark to slightly lighter)
for i in range(hero_h):
    t = i / max(1, hero_h-1)
    # interpolate between hero_top and hero_bottom
    r = int(hero_top[0]*(1-t) + hero_bottom[0]*t)
    g = int(hero_top[1]*(1-t) + hero_bottom[1]*t)
    b = int(hero_top[2]*(1-t) + hero_bottom[2]*t)
    draw.line([(0, hero_top_y + i),(W, hero_top_y + i)], fill=(r,g,b))

# Hero bottom progress bar area (thin series of segments)
pb_y = hero_bottom_y - 18
seg_w = 110
gap = 16
start_x = 80
# base segments (muted gray)
for i in range(8):
    x1 = start_x + i * (seg_w + gap)
    x2 = x1 + seg_w
    draw.rounded_rectangle([(x1, pb_y),(x2, pb_y+6)], radius=3, fill=(120,120,120))
# active segment (white)
active_x1 = start_x
draw.rounded_rectangle([(active_x1, pb_y-1),(active_x1 + seg_w, pb_y+7)], radius=4, fill=(255,255,255))

# Subtle overlay gradient at top of content below hero
overlay_y = hero_bottom_y
draw.rectangle([(0, overlay_y),(W, overlay_y+6)], fill=(245,245,247))

# MAIN CONTENT BACKGROUND (remains white)
content_top = hero_bottom_y + 20

# Organizer card (rounded rectangle behind organizer info)
org_card_top = 980
org_card_left = 36
org_card_right = W - 36
org_card_height = 120
org_card_box = (org_card_left, org_card_top, org_card_right, org_card_top + org_card_height)
draw.rounded_rectangle(org_card_box, radius=20, fill=card_fill, outline=card_border, width=1)

# Slight inner shadow for organizer card (subtle)
draw.line([(org_card_left+2, org_card_top+2),(org_card_right-2, org_card_top+2)], fill=(245,243,246), width=1)

# Separator line between organizer and details area
sep_y = org_card_top + org_card_height + 40
draw.line([(40, sep_y),(W-40, sep_y)], fill=divider_gray, width=1)

# Small details area separators (icons/labels will be pasted later)
details_top = sep_y + 30
# Draw light icon-row divider
draw.line([(40, details_top+150),(W-40, details_top+150)], fill=divider_gray, width=1)

# "Select date and time" cards row (background cards)
date_row_top = 1760
card_w = 420
card_h = 420 - 160  # visual height for the small date cards
card_spacing = 36
x = 36
y = date_row_top
for i in range(4):  # draw several placeholders across horizontally
    box = (x, y, x + card_w, y + card_h)
    # fill white with very light border
    draw.rounded_rectangle(box, radius=18, fill=(255,255,255), outline=(235,235,245), width=2)
    # inner faint underline to suggest selection indicator
    draw.line([(x+20, y+55),(x+card_w-20, y+55)], fill=(245,245,250), width=2)
    x += card_w + card_spacing

# Ticket selection card (rounded box with accent border)
ticket_card_top = 2100
ticket_card_box = (36, ticket_card_top, W-36, ticket_card_top + 160)
draw.rounded_rectangle(ticket_card_box, radius=18, fill=(255,255,255), outline=accent_blue, width=6)

# subtle inner divider inside ticket card
draw.line([(ticket_card_box[0]+24, ticket_card_top+72),(ticket_card_box[2]-24, ticket_card_top+72)], fill=(245,245,247), width=1)

# Drawer shadow under ticket card
shadow_y = ticket_card_top + 170
draw.rectangle([(ticket_card_box[0]+6, shadow_y),(ticket_card_box[2]-6, shadow_y+8)], fill=(240,240,241))

# Bottom checkout bar background (will be overlaid by detected checkout element)
checkout_bar_top = 2324
checkout_bar_height = 180
checkout_box = (0, checkout_bar_top, W, checkout_bar_top + checkout_bar_height)
# big orange bar background
draw.rectangle(checkout_box, fill=accent_orange)

# Thin top separator above checkout bar
draw.line([(20, checkout_bar_top),(W-20, checkout_bar_top)], fill=(190,60,30), width=3)

# Edge decorations and final subtle dividers
# small horizontal divider under header content
draw.line([(36, 880),(W-36, 880)], fill=divider_gray, width=1)

# Left margin vertical guide (visual, very faint)
draw.line([(36, hero_bottom_y+12),(36, H-200)], fill=(250,250,252), width=1)

# Right margin vertical guide (very faint)
draw.line([(W-36, hero_bottom_y+12),(W-36, H-200)], fill=(250,250,252), width=1)

# Add a faint bottom page fade to suggest depth
fade_top = H - 120
for i in range(60):
    alpha = int(10 * (1 - i/60))
    y = fade_top + i
    if y < H:
        draw.line([(0,y),(W,y)], fill=(255,255,255,alpha))

# Done - structural elements drawn. Content (icons, text, buttons) will be pasted on top.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1068), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1068, 1344, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/01_icon_April.png
try:
    _c1 = get_crop(1, 450, 352)
    canvas.paste(_c1, (24, 1972), _c1)
except Exception:
    pass
layout["April"] = [24, 1972, 474, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/02_icon_April.png
try:
    _c2 = get_crop(2, 450, 352)
    canvas.paste(_c2, (474, 1972), _c2)
except Exception:
    pass
layout["April"] = [474, 1972, 924, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/03_icon_27.png
try:
    _c3 = get_crop(3, 111, 104)
    canvas.paste(_c3, (988, 2440), _c3)
except Exception:
    pass
layout["27"] = [988, 2440, 1099, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/04_icon_Check_out_for_S35.00.png
try:
    _c4 = get_crop(4, 1440, 636)
    canvas.paste(_c4, (0, 2324), _c4)
except Exception:
    pass
layout["Check_out_for_S35.00"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/05_icon_27.png
try:
    _c5 = get_crop(5, 450, 352)
    canvas.paste(_c5, (924, 1972), _c5)
except Exception:
    pass
layout["27"] = [924, 1972, 1374, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/06_icon_7.38.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (36, 108), _c6)
except Exception:
    pass
layout["7.38"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/07_icon_27.png
try:
    _c7 = get_crop(7, 108, 104)
    canvas.paste(_c7, (1215, 2441), _c7)
except Exception:
    pass
layout["27"] = [1215, 2441, 1323, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/08_icon_April.png
try:
    _c8 = get_crop(8, 450, 352)
    canvas.paste(_c8, (924, 1972), _c8)
except Exception:
    pass
layout["April"] = [924, 1972, 1374, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/09_icon_27.png
try:
    _c9 = get_crop(9, 90, 99)
    canvas.paste(_c9, (1109, 2444), _c9)
except Exception:
    pass
layout["27"] = [1109, 2444, 1199, 2543]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/10_icon_Cha.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1260, 108), _c10)
except Exception:
    pass
layout["Cha"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/11_icon_Kangalee_Arts_Ensemble_Inc..png
try:
    _c11 = get_crop(11, 629, 144)
    canvas.paste(_c11, (288, 1028), _c11)
except Exception:
    pass
layout["Kangalee_Arts_Ensemble,_I"] = [288, 1028, 917, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 55, 64)
    canvas.paste(_c12, (1317, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [1317, 1, 1372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/13_icon_7.38.png
try:
    _c13 = get_crop(13, 63, 63)
    canvas.paste(_c13, (180, 1), _c13)
except Exception:
    pass
layout["7.38"] = [180, 1, 243, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/14_icon_S35.00.png
try:
    _c14 = get_crop(14, 99, 103)
    canvas.paste(_c14, (291, 2576), _c14)
except Exception:
    pass
layout["S35.00"] = [291, 2576, 390, 2679]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/15_icon_26.png
try:
    _c15 = get_crop(15, 450, 352)
    canvas.paste(_c15, (474, 1972), _c15)
except Exception:
    pass
layout["26"] = [474, 1972, 924, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 52, 62)
    canvas.paste(_c16, (1262, 2), _c16)
except Exception:
    pass
layout["icon_16"] = [1262, 2, 1314, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/17_icon_7.38.png
try:
    _c17 = get_crop(17, 60, 65)
    canvas.paste(_c17, (115, 0), _c17)
except Exception:
    pass
layout["7.38"] = [115, 0, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 62, 61)
    canvas.paste(_c18, (1216, 3), _c18)
except Exception:
    pass
layout["icon_18"] = [1216, 3, 1278, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/19_icon_Thursday_April_25.png
try:
    _c19 = get_crop(19, 181, 58)
    canvas.paste(_c19, (252, 538), _c19)
except Exception:
    pass
layout["Thursday_April_25"] = [252, 538, 433, 596]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 54, 62)
    canvas.paste(_c20, (247, 2), _c20)
except Exception:
    pass
layout["icon_20"] = [247, 2, 301, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 59, 62)
    canvas.paste(_c21, (311, 2), _c21)
except Exception:
    pass
layout["icon_21"] = [311, 2, 370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 50, 66)
    canvas.paste(_c22, (382, 1), _c22)
except Exception:
    pass
layout["icon_22"] = [382, 1, 432, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/23_icon_2_hrs_30_mins.png
try:
    _c23 = get_crop(23, 294, 72)
    canvas.paste(_c23, (135, 1441), _c23)
except Exception:
    pass
layout["2_hrs_30_mins"] = [135, 1441, 429, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/24_icon_S35.00.png
try:
    _c24 = get_crop(24, 182, 95)
    canvas.paste(_c24, (109, 2575), _c24)
except Exception:
    pass
layout["S35.00"] = [109, 2575, 291, 2670]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/25_text_7.38.png
try:
    _c25 = get_crop(25, 92, 43)
    canvas.paste(_c25, (22, 17), _c25)
except Exception:
    pass
layout["7.38"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/26_text_Cha.png
try:
    _c26 = get_crop(26, 37, 18)
    canvas.paste(_c26, (1245, 288), _c26)
except Exception:
    pass
layout["Cha"] = [1245, 288, 1282, 306]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/27_text_Thursday_April_25.png
try:
    _c27 = get_crop(27, 456, 77)
    canvas.paste(_c27, (40, 758), _c27)
except Exception:
    pass
layout["Thursday_April_25"] = [40, 758, 496, 835]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/28_text_8_00_PM.png
try:
    _c28 = get_crop(28, 215, 63)
    canvas.paste(_c28, (520, 762), _c28)
except Exception:
    pass
layout["8:00_PM"] = [520, 762, 735, 825]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/29_text_THE_LIFE_DEATH_OF_ART.png
try:
    _c29 = get_crop(29, 629, 144)
    canvas.paste(_c29, (288, 1028), _c29)
except Exception:
    pass
layout["THE_LIFE_&_DEATH_OF_ART"] = [288, 1028, 917, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/30_text_JACK.png
try:
    _c30 = get_crop(30, 121, 52)
    canvas.paste(_c30, (137, 1341), _c30)
except Exception:
    pass
layout["JACK"] = [137, 1341, 258, 1393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/31_text_Refund_policy.png
try:
    _c31 = get_crop(31, 299, 63)
    canvas.paste(_c31, (138, 1558), _c31)
except Exception:
    pass
layout["Refund_policy"] = [138, 1558, 437, 1621]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/32_text_The_organizer_will_review_refund_request.png
try:
    _c32 = get_crop(32, 1344, 144)
    canvas.paste(_c32, (48, 1295), _c32)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/33_text_Select_date_and_time.png
try:
    _c33 = get_crop(33, 450, 352)
    canvas.paste(_c33, (24, 1972), _c33)
except Exception:
    pass
layout["Select_date_and_time"] = [24, 1972, 474, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/34_text_General_Admission.png
try:
    _c34 = get_crop(34, 415, 55)
    canvas.paste(_c34, (116, 2451), _c34)
except Exception:
    pass
layout["General_Admission"] = [116, 2451, 531, 2506]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/35_clickable_More.png
try:
    _c35 = get_crop(35, 144, 144)
    canvas.paste(_c35, (1116, 108), _c35)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_03_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-5/36_clickable_Organizer_profile_picture.png
try:
    _c36 = get_crop(36, 144, 144)
    canvas.paste(_c36, (96, 1067), _c36)
except Exception:
    pass
layout["Organizer_profile_picture"] = [96, 1067, 240, 1211]
