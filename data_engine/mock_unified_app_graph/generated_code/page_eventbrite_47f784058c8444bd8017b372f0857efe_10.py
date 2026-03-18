# page_id: page_eventbrite_47f784058c8444bd8017b372f0857efe_10
# screenshot: 2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12.png
# step_index: 10/11
# task: Open Eventbrite. Explore local events scheduled for this weekend. Select the first event from the 'Science' category. Read details of the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for a 1440x2960 mobile UI using PIL draw (canvas, draw provided)

# Colors (approximate to screenshot)
BG = (250, 249, 252)           # overall page background (very light)
STATUS_BAR = (120, 120, 120)   # status bar dark gray
HERO_BG = (230, 230, 232)      # hero image background placeholder
CARD_BG = (255, 255, 255)      # main white card
CARD_SHADOW = (235, 235, 240)  # subtle shadow / stroke for card
DIVIDER = (230, 229, 234)      # thin separator lines
BADGE_BG = (253, 236, 238)     # "Going fast" badge background (light pink)
BADGE_BORDER = (246, 213, 217)
PILL_BG = (244, 247, 250)      # category pill background
PILL_BORDER = (222, 227, 232)
BOTTOM_BAR = (249, 248, 250)   # sticky bottom bar background
GET_BTN = (212, 83, 30)        # orange "Get tickets" button background
GET_BTN_SHADOW = (190, 70, 26)

w, h = canvas.size

# 1) Overall background
draw.rectangle([(0, 0), (w, h)], fill=BG)

# 2) Status bar area at top (~72px)
status_h = 72
draw.rectangle([(0, 0), (w, status_h)], fill=STATUS_BAR)

# subtle separator under status bar
draw.line([(0, status_h), (w, status_h)], fill=(0,0,0,20), width=1)

# 3) Hero image background area (placeholder block behind the hero photo)
hero_top = status_h
hero_bottom = 700
draw.rectangle([(0, hero_top), (w, hero_bottom)], fill=HERO_BG)

# Add a soft darker band at the very top of the hero to emulate overlay under status icons
draw.rectangle([(0, hero_top), (w, hero_top+28)], fill=(220, 220, 222))

# 4) Main content card (rounded white card overlapping hero)
card_radius = 36
card_top = hero_bottom - 60  # slight overlap
card_rect = [0, card_top, w, h - 280]  # leave space at bottom for sticky bar
draw.rounded_rectangle(card_rect, radius=card_radius, fill=CARD_BG, outline=CARD_SHADOW, width=1)

# subtle top shadow line for the card
draw.line([(24, card_top+2), (w-24, card_top+2)], fill=CARD_SHADOW, width=1)

# 5) "Going fast" badge background (do not draw icon/text)
# Use detected pos & size: pos=(41,753) size=334x86
badge_x, badge_y = 41, 753
badge_w, badge_h = 334, 86
badge_box = [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h]
draw.rounded_rectangle(badge_box, radius=40, fill=BADGE_BG, outline=BADGE_BORDER, width=1)

# 6) Horizontal divider under the location/refund policy area
# Place divider across card roughly where the small grey line appears in screenshot.
divider_y = 1560
draw.line([(48, divider_y), (w-48, divider_y)], fill=DIVIDER, width=2)

# 7) Category pill background under "About this event"
# Detected pos=(36,1927) size=675x97
pill_x, pill_y = 36, 1927
pill_w, pill_h = 675, 97
pill_box = [pill_x, pill_y, pill_x + pill_w, pill_y + pill_h]
draw.rounded_rectangle(pill_box, radius=48, fill=PILL_BG, outline=PILL_BORDER, width=1)

# 8) Additional subtle separators between sections inside the card
# small light lines between major content blocks
sep_positions = [1200, 1700, 2020]
for y_pos in sep_positions:
    draw.line([(48, y_pos), (w-48, y_pos)], fill=DIVIDER, width=1)

# 9) Sticky bottom bar with "Free" area and orange "Get tickets" button background
bottom_bar_top = h - 280
draw.rectangle([(0, bottom_bar_top), (w, h)], fill=BOTTOM_BAR)

# top divider for bottom bar
draw.line([(0, bottom_bar_top), (w, bottom_bar_top)], fill=DIVIDER, width=2)

# left area background (where "Free" label will be pasted) - keep it subtle and aligned
left_pad = 48
left_area = [left_pad, bottom_bar_top + 24, w//2 - 24, h - 24]
draw.rectangle(left_area, fill=(255,255,255,0))  # essentially transparent; keep structure minimal

# draw the orange "Get tickets" button background (use detected pos & size)
btn_x, btn_y = 822, 2768
btn_w, btn_h = 570, 144
btn_box = [btn_x, btn_y, btn_x + btn_w, btn_y + btn_h]
draw.rounded_rectangle(btn_box, radius=12, fill=GET_BTN, outline=GET_BTN_SHADOW, width=0)

# subtle drop shadow under the button (a darker stripe)
shadow_box = [btn_x+6, btn_y+btn_h-6, btn_x+btn_w-6, btn_y+btn_h]
draw.rectangle(shadow_box, fill=GET_BTN_SHADOW)

# 10) Small accent: faint left/right margins shadow for content card bottom
shadow_y0 = card_rect[3] - 8
draw.line([(24, shadow_y0), (w-24, shadow_y0)], fill=(240, 240, 243), width=2)

# End of structural drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/00_icon_Get_tickets.png
try:
    _c0 = get_crop(0, 570, 144)
    canvas.paste(_c0, (822, 2768), _c0)
except Exception:
    pass
layout["Get_tickets"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/01_icon_More.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1116, 108), _c1)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/02_icon_Share.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1260, 108), _c2)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/03_icon_Going_fast.png
try:
    _c3 = get_crop(3, 334, 86)
    canvas.paste(_c3, (41, 753), _c3)
except Exception:
    pass
layout["Going_fast"] = [41, 753, 375, 839]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/04_icon_Science_Technology.png
try:
    _c4 = get_crop(4, 675, 97)
    canvas.paste(_c4, (36, 1927), _c4)
except Exception:
    pass
layout["Science_&_Technology"] = [36, 1927, 711, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/05_icon_7.59.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (36, 108), _c5)
except Exception:
    pass
layout["7.59"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/06_icon_7.59.png
try:
    _c6 = get_crop(6, 65, 70)
    canvas.paste(_c6, (179, 1), _c6)
except Exception:
    pass
layout["7.59"] = [179, 1, 244, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/07_icon_7.59.png
try:
    _c7 = get_crop(7, 62, 70)
    canvas.paste(_c7, (114, 0), _c7)
except Exception:
    pass
layout["7.59"] = [114, 0, 176, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 54, 61)
    canvas.paste(_c8, (1318, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [1318, 2, 1372, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 56, 67)
    canvas.paste(_c9, (247, 2), _c9)
except Exception:
    pass
layout["icon_9"] = [247, 2, 303, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 65, 68)
    canvas.paste(_c10, (309, 2), _c10)
except Exception:
    pass
layout["icon_10"] = [309, 2, 374, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 57, 59)
    canvas.paste(_c11, (1217, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [1217, 3, 1274, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 43, 57)
    canvas.paste(_c12, (1272, 5), _c12)
except Exception:
    pass
layout["icon_12"] = [1272, 5, 1315, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/13_icon_9_30AM.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1116, 108), _c13)
except Exception:
    pass
layout["9:30AM"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 52, 67)
    canvas.paste(_c14, (382, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [382, 2, 434, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/15_text_7.59.png
try:
    _c15 = get_crop(15, 89, 43)
    canvas.paste(_c15, (22, 17), _c15)
except Exception:
    pass
layout["7.59"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/16_text_Area_Bioengineering_Symposium.png
try:
    _c16 = get_crop(16, 1344, 144)
    canvas.paste(_c16, (48, 1277), _c16)
except Exception:
    pass
layout["Area_Bioengineering_Sympo"] = [48, 1277, 1392, 1421]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/17_text_BABS.png
try:
    _c17 = get_crop(17, 245, 84)
    canvas.paste(_c17, (46, 1109), _c17)
except Exception:
    pass
layout["[BABS]"] = [46, 1109, 291, 1193]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/18_text_Hearst_Memorial_Mining_Building.png
try:
    _c18 = get_crop(18, 1344, 144)
    canvas.paste(_c18, (48, 1277), _c18)
except Exception:
    pass
layout["Hearst_Memorial_Mining_Bu"] = [48, 1277, 1392, 1421]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/19_text_7_hrs_30_mins.png
try:
    _c19 = get_crop(19, 290, 54)
    canvas.paste(_c19, (141, 1432), _c19)
except Exception:
    pass
layout["7_hrs_30_mins"] = [141, 1432, 431, 1486]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/20_text_Refund_policy.png
try:
    _c20 = get_crop(20, 299, 63)
    canvas.paste(_c20, (138, 1539), _c20)
except Exception:
    pass
layout["Refund_policy"] = [138, 1539, 437, 1602]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/21_text_The_organizer_will_review_refund_request.png
try:
    _c21 = get_crop(21, 1344, 144)
    canvas.paste(_c21, (48, 1277), _c21)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1277, 1392, 1421]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/22_text_About_this_event.png
try:
    _c22 = get_crop(22, 454, 61)
    canvas.paste(_c22, (45, 1840), _c22)
except Exception:
    pass
layout["About_this_event"] = [45, 1840, 499, 1901]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/23_text_Join_us_for_an_in-depth_dive_into_the_bi.png
try:
    _c23 = get_crop(23, 1344, 144)
    canvas.paste(_c23, (48, 1277), _c23)
except Exception:
    pass
layout["Join_us_for_an_in-depth_d"] = [48, 1277, 1392, 1421]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/24_text_Areal.png
try:
    _c24 = get_crop(24, 119, 51)
    canvas.paste(_c24, (742, 2142), _c24)
except Exception:
    pass
layout["Areal"] = [742, 2142, 861, 2193]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/25_text_Welcome_to_the.png
try:
    _c25 = get_crop(25, 336, 52)
    canvas.paste(_c25, (44, 2266), _c25)
except Exception:
    pass
layout["Welcome_to_the"] = [44, 2266, 380, 2318]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/26_text_Area_Bioengineering_Symposiuml.png
try:
    _c26 = get_crop(26, 711, 64)
    canvas.paste(_c26, (466, 2265), _c26)
except Exception:
    pass
layout["Area_Bioengineering_Sympo"] = [466, 2265, 1177, 2329]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/27_text_Date_Saturday_April_27th_2024.png
try:
    _c27 = get_crop(27, 688, 67)
    canvas.paste(_c27, (45, 2389), _c27)
except Exception:
    pass
layout["Date:_Saturday,_April_27t"] = [45, 2389, 733, 2456]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/28_text_Location_Hearst_Memorial_Mining_Building.png
try:
    _c28 = get_crop(28, 570, 144)
    canvas.paste(_c28, (822, 2768), _c28)
except Exception:
    pass
layout["Location:_Hearst_Memorial"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/29_text_Join_us_for_a.png
try:
    _c29 = get_crop(29, 254, 52)
    canvas.paste(_c29, (39, 2646), _c29)
except Exception:
    pass
layout["Join_us_for_a"] = [39, 2646, 293, 2698]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/30_text_filled_with_exciting_talks_poster_presen.png
try:
    _c30 = get_crop(30, 570, 144)
    canvas.paste(_c30, (822, 2768), _c30)
except Exception:
    pass
layout["filled_with_exciting_talk"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_10_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-12/31_text_Free.png
try:
    _c31 = get_crop(31, 110, 55)
    canvas.paste(_c31, (89, 2816), _c31)
except Exception:
    pass
layout["Free"] = [89, 2816, 199, 2871]
