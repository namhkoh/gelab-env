# page_id: page_eventbrite_39adaf730c584c5582b89d1335e0c2cd_06
# screenshot: 2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8.png
# step_index: 6/6
# task: Open Eventbrite. Search for 'food and drink' events. Follow the organizer of the first event in listing.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background base
draw.rectangle((0, 0, 1440, 2960), fill='#FFFFFF')

# Status bar (top)
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill='#E6E6E6')

# Header / Hero image area (dark image background)
hero_top = status_h
hero_bottom = 520
# subtle vertical gradient simulation
for i in range(hero_top, hero_bottom):
    t = (i - hero_top) / max(1, (hero_bottom - hero_top))
    # interpolate between two dark tones
    r = int(40 + (18 - 40) * t)
    g = int(46 + (18 - 46) * t)
    b = int(52 + (18 - 52) * t)
    draw.line([(0, i), (1440, i)], fill=(r, g, b))
# Soft darker overlay at bottom of hero to hint separation
draw.rectangle((0, hero_bottom - 36, 1440, hero_bottom), fill=(0, 0, 0, 20))

# Progress bar overlay near bottom of hero image
prog_y = hero_bottom - 18
bar_margin = 72
bar_h = 8
# background rail
draw.rounded_rectangle((bar_margin, prog_y, 1440 - bar_margin, prog_y + bar_h),
                       radius=6, fill='#BFBFBF')
# progress fill (shorter)
draw.rounded_rectangle((bar_margin, prog_y, int((1440 - 2*bar_margin) * 0.42) + bar_margin, prog_y + bar_h),
                       radius=6, fill='#FFFFFF')

# Yellow "pill" status banner background (no icon/text)
pill_left = 48
pill_right = 1440 - 48
pill_top = hero_bottom + 36
pill_bottom = pill_top + 72
draw.rounded_rectangle((pill_left, pill_top, pill_right, pill_bottom),
                       radius=16, fill='#FFF1A8', outline=None)

# Main content area separation shadow line under pill
sep_y = pill_bottom + 34
draw.line((48, sep_y, 1440 - 48, sep_y), fill='#EFECEF', width=1)

# Organizer card background (rounded)
card_left = 48
card_right = 1440 - 48
card_top = 1120
card_bottom = card_top + 168
draw.rounded_rectangle((card_left, card_top, card_right, card_bottom),
                       radius=20, fill='#F7F6F9')

# Subtle inner top highlight on organizer card
draw.line((card_left + 8, card_top + 6, card_right - 8, card_top + 6), fill='#FFFFFF', width=1)

# Thin divider line under the organizer/summary area
divider_y = card_bottom + 180
draw.line((48, divider_y, 1440 - 48, divider_y), fill='#ECE8EE', width=2)

# Refund/Info area background stays white; add a light separator under it
info_bottom = divider_y + 120
draw.line((48, info_bottom, 1440 - 48, info_bottom), fill='#F0EEF2', width=1)

# "Select date and time" cards row backgrounds
cards_top = info_bottom + 48
card_w = 420
card_h = 380
card_spacing = 36
start_x = 48
# Three cards
for i in range(3):
    x0 = start_x + i * (card_w + card_spacing)
    x1 = x0 + card_w
    y0 = cards_top
    y1 = y0 + card_h
    # card background
    draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill='#FFFFFF', outline='#EFEFF4', width=4)
# Selected indicator for first card: blue border and inner circle (background only)
sel_x0 = start_x
sel_x1 = sel_x0 + card_w
draw.rounded_rectangle((sel_x0, cards_top, sel_x1, cards_top + card_h), radius=18,
                       outline='#3650FF', width=6, fill=None)
# big circular date background (behind number) in first card
circle_cx = sel_x0 + card_w // 2
circle_cy = cards_top + 160
circle_r = 58
draw.ellipse((circle_cx - circle_r, circle_cy - circle_r, circle_cx + circle_r, circle_cy + circle_r),
             fill='#3753FF')

# Light placeholder round backgrounds for other cards (grey faint circles)
for i in [1, 2]:
    cx = start_x + i * (card_w + card_spacing) + card_w // 2
    cy = cards_top + 160
    r = 46
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill='#F4F4F6')

# Subtle horizontal separator before bottom action bar
action_bar_top = 2756
draw.line((0, action_bar_top, 1440, action_bar_top), fill='#E9E7EA', width=2)

# Bottom action bar background
draw.rectangle((0, action_bar_top, 1440, 2960), fill='#F6F4F7')

# Left "Sales ended" pill area background (button background only, no text)
left_btn_left = 48
left_btn_right = 660
left_btn_top = action_bar_top + 24
left_btn_bottom = left_btn_top + 120
draw.rounded_rectangle((left_btn_left, left_btn_top, left_btn_right, left_btn_bottom),
                       radius=12, fill='#FFFFFF', outline=None)

# Right "Details" button background (outlined)
right_btn_left = 760
right_btn_right = 1392
right_btn_top = left_btn_top
right_btn_bottom = left_btn_bottom
draw.rounded_rectangle((right_btn_left, right_btn_top, right_btn_right, right_btn_bottom),
                       radius=12, fill='#FFFFFF', outline='#9D98A2', width=6)

# Final subtle vertical dividers and spacing marks (structure only)
# small thin divider above organizer card to separate from content
draw.line((48, card_top - 28, 1440 - 48, card_top - 28), fill='#F2F0F3', width=1)

# end of structural drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/00_icon_Following.png
try:
    _c0 = get_crop(0, 398, 144)
    canvas.paste(_c0, (946, 1236), _c0)
except Exception:
    pass
layout["Following"] = [946, 1236, 1344, 1380]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/01_icon_Details.png
try:
    _c1 = get_crop(1, 522, 144)
    canvas.paste(_c1, (822, 2768), _c1)
except Exception:
    pass
layout["Details"] = [822, 2768, 1344, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/02_icon_More.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1116, 108), _c2)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/03_icon_23.png
try:
    _c3 = get_crop(3, 450, 516)
    canvas.paste(_c3, (24, 2140), _c3)
except Exception:
    pass
layout["23"] = [24, 2140, 474, 2656]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/04_icon_Share.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1260, 108), _c4)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/05_icon_7.45_my.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (36, 108), _c5)
except Exception:
    pass
layout["7.45_my"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/06_icon_Sales_ended.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1116, 108), _c6)
except Exception:
    pass
layout["Sales_ended"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/07_icon_25.png
try:
    _c7 = get_crop(7, 450, 516)
    canvas.paste(_c7, (924, 2140), _c7)
except Exception:
    pass
layout["25"] = [924, 2140, 1374, 2656]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/08_icon_24.png
try:
    _c8 = get_crop(8, 450, 516)
    canvas.paste(_c8, (474, 2140), _c8)
except Exception:
    pass
layout["24"] = [474, 2140, 924, 2656]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/09_icon_5.30_PM.png
try:
    _c9 = get_crop(9, 291, 144)
    canvas.paste(_c9, (288, 1196), _c9)
except Exception:
    pass
layout["5.30_PM"] = [288, 1196, 579, 1340]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/10_icon_7.45_my.png
try:
    _c10 = get_crop(10, 61, 70)
    canvas.paste(_c10, (181, 0), _c10)
except Exception:
    pass
layout["7.45_my"] = [181, 0, 242, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 69, 72)
    canvas.paste(_c11, (307, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [307, 1, 376, 73]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 97, 66)
    canvas.paste(_c12, (1212, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1212, 0, 1309, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 60, 67)
    canvas.paste(_c13, (1315, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1315, 0, 1375, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 58, 70)
    canvas.paste(_c14, (246, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [246, 1, 304, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 52, 73)
    canvas.paste(_c15, (381, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [381, 0, 433, 73]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/16_icon_5.30_PM.png
try:
    _c16 = get_crop(16, 291, 144)
    canvas.paste(_c16, (288, 1196), _c16)
except Exception:
    pass
layout["5.30_PM"] = [288, 1196, 579, 1340]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/17_text_7.45_my.png
try:
    _c17 = get_crop(17, 149, 43)
    canvas.paste(_c17, (22, 15), _c17)
except Exception:
    pass
layout["7.45_my"] = [22, 15, 171, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/18_text_7_KINGDOMS.png
try:
    _c18 = get_crop(18, 291, 144)
    canvas.paste(_c18, (288, 1196), _c18)
except Exception:
    pass
layout["7_KINGDOMS"] = [288, 1196, 579, 1340]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/19_text_NGDOM.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (96, 1235), _c19)
except Exception:
    pass
layout["NGDOM"] = [96, 1235, 240, 1379]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/20_text_366_Followers.png
try:
    _c20 = get_crop(20, 291, 144)
    canvas.paste(_c20, (288, 1196), _c20)
except Exception:
    pass
layout["366_Followers"] = [288, 1196, 579, 1340]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/21_text_Kingdoms.png
try:
    _c21 = get_crop(21, 227, 66)
    canvas.paste(_c21, (174, 1509), _c21)
except Exception:
    pass
layout["Kingdoms"] = [174, 1509, 401, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/22_text_2_hrs.png
try:
    _c22 = get_crop(22, 112, 50)
    canvas.paste(_c22, (141, 1621), _c22)
except Exception:
    pass
layout["2_hrs"] = [141, 1621, 253, 1671]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/23_text_Refund_policy.png
try:
    _c23 = get_crop(23, 299, 60)
    canvas.paste(_c23, (138, 1727), _c23)
except Exception:
    pass
layout["Refund_policy"] = [138, 1727, 437, 1787]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/24_text_The_organizer_will_review_refund_request.png
try:
    _c24 = get_crop(24, 1344, 144)
    canvas.paste(_c24, (48, 1463), _c24)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1463, 1392, 1607]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/25_text_Select_date_and_time.png
try:
    _c25 = get_crop(25, 450, 516)
    canvas.paste(_c25, (24, 2140), _c25)
except Exception:
    pass
layout["Select_date_and_time"] = [24, 2140, 474, 2656]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/26_text_Thursday.png
try:
    _c26 = get_crop(26, 191, 63)
    canvas.paste(_c26, (1052, 2211), _c26)
except Exception:
    pass
layout["Thursday"] = [1052, 2211, 1243, 2274]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/27_text_April.png
try:
    _c27 = get_crop(27, 450, 516)
    canvas.paste(_c27, (924, 2140), _c27)
except Exception:
    pass
layout["April"] = [924, 2140, 1374, 2656]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_06_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-8/28_text_Sales_ended.png
try:
    _c28 = get_crop(28, 274, 55)
    canvas.paste(_c28, (90, 2814), _c28)
except Exception:
    pass
layout["Sales_ended"] = [90, 2814, 364, 2869]
