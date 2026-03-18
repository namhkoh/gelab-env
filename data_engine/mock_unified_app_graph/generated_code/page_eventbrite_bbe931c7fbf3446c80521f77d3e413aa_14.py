# page_id: page_eventbrite_bbe931c7fbf3446c80521f77d3e413aa_14
# screenshot: 2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16.png
# step_index: 14/20
# task: Open Eventbrite. Search free events in Los Angeles. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the Eventbrite-like page
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Full canvas background (very light, slight cool tint to match screenshot)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFBFD")

# Status bar area at top (~50-100px). Keep neutral gray background
status_h = 96
draw.rectangle([(0, 0), (1440, status_h)], fill="#CFCFCF")
# subtle inner top highlight
draw.line([(0, status_h - 1), (1440, status_h - 1)], fill="#BDBDBD")

# Header / toolbar area (white) directly under status bar
header_h = 128
header_y0 = status_h
header_y1 = header_y0 + header_h
draw.rectangle([(0, header_y0), (1440, header_y1)], fill="#FFFFFF")
# subtle divider/shadow under header
draw.line([(24, header_y1), (1416, header_y1)], fill="#E6E6EA", width=1)
draw.line([(24, header_y1 + 2), (1416, header_y1 + 2)], fill="#F3F3F5", width=1)

# Main subtle horizontal separator (separates top info area from content)
sep1_y = header_y1 + 196  # approx under refund/policy area
draw.line([(24, sep1_y), (1416, sep1_y)], fill="#F0F0F2", width=1)

# "About this event" section area background (keeps white overall)
# Draw a larger faint separator slightly above the section title area
about_sep_y = header_y1 + 420
draw.line([(24, about_sep_y), (1416, about_sep_y)], fill="#F0F0F2", width=1)

# Pill tag background for category (rounded pill)
pill_left = 72
pill_top = 762
pill_right = pill_left + 948
pill_bottom = pill_top + 64
draw.rounded_rectangle(
    [(pill_left, pill_top), (pill_right, pill_bottom)],
    radius=36,
    fill="#EEF3FB",
    outline=None,
)

# underline / subtle divider before Location/ticket area
loc_div_y = 2220
draw.line([(24, loc_div_y), (1416, loc_div_y)], fill="#E9E9EE", width=1)

# Ticket selection card (rounded rectangle with blue outline)
card_left = 48
card_top = 2350
card_right = 1392
card_bottom = 2640
# outer subtle shadow rectangle (light)
shadow_margin = 6
draw.rounded_rectangle(
    [(card_left + shadow_margin, card_top + shadow_margin), (card_right + shadow_margin, card_bottom + shadow_margin)],
    radius=24,
    fill="#F8F8F9",
    outline=None,
)
# main card (white fill with blue border)
draw.rounded_rectangle(
    [(card_left, card_top), (card_right, card_bottom)],
    radius=20,
    fill="#FFFFFF",
    outline="#2C53FF",
    width=6,
)

# Inner divider inside the ticket card to separate title area from controls (light)
inner_div_y = card_top + 120
draw.line([(card_left + 24, inner_div_y), (card_right - 24, inner_div_y)], fill="#F0F2F8", width=1)

# Small faint rounded container behind price label (left side) to hint background (do not draw text)
price_box_left = card_left + 28
price_box_top = card_top + 118
price_box_right = price_box_left + 220
price_box_bottom = price_box_top + 80
draw.rounded_rectangle(
    [(price_box_left, price_box_top), (price_box_right, price_box_bottom)],
    radius=12,
    fill="#FFFFFF",
    outline=None,
)

# Large subtle horizontal separator above bottom area (before reserve button)
bottom_sep_y = card_bottom + 40
draw.line([(24, bottom_sep_y), (1416, bottom_sep_y)], fill="#F0F0F2", width=1)

# Decorative thin left content guide line (visual structure, faint)
draw.line([(72, header_y1 + 12), (72, 2600)], fill="#F4F4F6", width=2)

# Add a subtle vertical left margin indicator stripe (purely structural)
draw.rectangle([(0, header_y1), (8, 2600)], fill="#FBFBFD")

# Final subtle footer area background (slightly warmer to suggest interactive zone)
footer_top = 2736
draw.rectangle([(0, footer_top), (1440, 2960)], fill="#FFFFFF")
# (Do not draw the Reserve button itself; it will be pasted later.)

# Done - structural elements drawn. (No icons or text have been drawn.)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/00_icon_More.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1116, 108), _c0)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/01_icon_Share.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1260, 108), _c1)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/02_icon_hrs.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (36, 108), _c2)
except Exception:
    pass
layout["hrs"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/03_icon_9.13.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (36, 108), _c3)
except Exception:
    pass
layout["9.13"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/04_icon_Decrease.png
try:
    _c4 = get_crop(4, 99, 96)
    canvas.paste(_c4, (996, 2444), _c4)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 51, 55)
    canvas.paste(_c5, (316, 7), _c5)
except Exception:
    pass
layout["icon_5"] = [316, 7, 367, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/06_icon_Increase.png
try:
    _c6 = get_crop(6, 96, 96)
    canvas.paste(_c6, (1224, 2444), _c6)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 51, 55)
    canvas.paste(_c7, (249, 6), _c7)
except Exception:
    pass
layout["icon_7"] = [249, 6, 300, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 93, 104)
    canvas.paste(_c8, (1108, 2441), _c8)
except Exception:
    pass
layout["icon_8"] = [1108, 2441, 1201, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/09_icon_Reserve_a_spot.png
try:
    _c9 = get_crop(9, 1296, 132)
    canvas.paste(_c9, (72, 2756), _c9)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 54, 55)
    canvas.paste(_c10, (181, 6), _c10)
except Exception:
    pass
layout["icon_10"] = [181, 6, 235, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 43, 62)
    canvas.paste(_c11, (1157, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [1157, 3, 1200, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 97, 60)
    canvas.paste(_c12, (1215, 3), _c12)
except Exception:
    pass
layout["icon_12"] = [1215, 3, 1312, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 44, 56)
    canvas.paste(_c13, (1326, 5), _c13)
except Exception:
    pass
layout["icon_13"] = [1326, 5, 1370, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/14_icon_9.13.png
try:
    _c14 = get_crop(14, 52, 57)
    canvas.paste(_c14, (117, 5), _c14)
except Exception:
    pass
layout["9.13"] = [117, 5, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/15_icon_Free.png
try:
    _c15 = get_crop(15, 140, 111)
    canvas.paste(_c15, (96, 2569), _c15)
except Exception:
    pass
layout["Free"] = [96, 2569, 236, 2680]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 43, 56)
    canvas.paste(_c16, (386, 6), _c16)
except Exception:
    pass
layout["icon_16"] = [386, 6, 429, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/17_icon_Refund_policy.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (36, 108), _c17)
except Exception:
    pass
layout["Refund_policy"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/18_icon_Free.png
try:
    _c18 = get_crop(18, 75, 72)
    canvas.paste(_c18, (249, 2588), _c18)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/19_text_9.13.png
try:
    _c19 = get_crop(19, 91, 43)
    canvas.paste(_c19, (20, 17), _c19)
except Exception:
    pass
layout["9.13"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/20_text_Minorities_Building_..png
try:
    _c20 = get_crop(20, 556, 79)
    canvas.paste(_c20, (250, 150), _c20)
except Exception:
    pass
layout["Minorities_Building_."] = [250, 150, 806, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/21_text_The_organizer_will_review_refund_request.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1116, 108), _c21)
except Exception:
    pass
layout["The_organizer_will_review"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/22_text_About_this_event.png
try:
    _c22 = get_crop(22, 450, 57)
    canvas.paste(_c22, (46, 682), _c22)
except Exception:
    pass
layout["About_this_event"] = [46, 682, 496, 739]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/23_text_Business_Professional_._Startups_Small_B.png
try:
    _c23 = get_crop(23, 948, 56)
    canvas.paste(_c23, (87, 793), _c23)
except Exception:
    pass
layout["Business_&_Professional_."] = [87, 793, 1035, 849]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/24_text_We_invite_you_to_a_value-packed_educatio.png
try:
    _c24 = get_crop(24, 144, 144)
    canvas.paste(_c24, (1116, 108), _c24)
except Exception:
    pass
layout["We_invite_you_to_a_value-"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/25_text_Minorities_in_Franchising_Learn_How_to.png
try:
    _c25 = get_crop(25, 1135, 84)
    canvas.paste(_c25, (46, 1109), _c25)
except Exception:
    pass
layout["Minorities_in_Franchising"] = [46, 1109, 1181, 1193]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/26_text_Build_Wealth_in_the_Franchise_Industry_a.png
try:
    _c26 = get_crop(26, 1286, 89)
    canvas.paste(_c26, (46, 1193), _c26)
except Exception:
    pass
layout["Build_Wealth_in_the_Franc"] = [46, 1193, 1332, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/27_text_Franchise_Expo_West.png
try:
    _c27 = get_crop(27, 623, 90)
    canvas.paste(_c27, (43, 1285), _c27)
except Exception:
    pass
layout["Franchise_Expo_West"] = [43, 1285, 666, 1375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/28_text_We_invite_you_to_a_value-packed_educatio.png
try:
    _c28 = get_crop(28, 1292, 65)
    canvas.paste(_c28, (43, 1495), _c28)
except Exception:
    pass
layout["We_invite_you_to_a_value-"] = [43, 1495, 1335, 1560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/29_text_on_how_Minorities_can_profit_and_build_g.png
try:
    _c29 = get_crop(29, 99, 96)
    canvas.paste(_c29, (996, 2444), _c29)
except Exception:
    pass
layout["on_how_Minorities_can_pro"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/30_text_During_this_1_hour_session_you_will_lear.png
try:
    _c30 = get_crop(30, 99, 96)
    canvas.paste(_c30, (996, 2444), _c30)
except Exception:
    pass
layout["During_this_1_hour_sessio"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/31_text_affects_growth_access_to_capital_and_the.png
try:
    _c31 = get_crop(31, 99, 96)
    canvas.paste(_c31, (996, 2444), _c31)
except Exception:
    pass
layout["affects_growth,_access_to"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/32_text_environment..png
try:
    _c32 = get_crop(32, 288, 49)
    canvas.paste(_c32, (42, 2003), _c32)
except Exception:
    pass
layout["environment."] = [42, 2003, 330, 2052]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/33_text_Date_Saturday_April_13th_2024at_1_OO_PM.png
try:
    _c33 = get_crop(33, 75, 72)
    canvas.paste(_c33, (249, 2588), _c33)
except Exception:
    pass
layout["Date:_Saturday;_April_13t"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/34_text_Location.png
try:
    _c34 = get_crop(34, 205, 52)
    canvas.paste(_c34, (44, 2255), _c34)
except Exception:
    pass
layout["Location:"] = [44, 2255, 249, 2307]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_14_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-16/35_text_Complimentary_Access.png
try:
    _c35 = get_crop(35, 75, 72)
    canvas.paste(_c35, (249, 2588), _c35)
except Exception:
    pass
layout["Complimentary_Access"] = [249, 2588, 324, 2660]
