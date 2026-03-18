# page_id: page_eventbrite_31d4c984d33c4fe69d15d4e595fa95f6_14
# screenshot: 2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16.png
# step_index: 14/14
# task: Open Eventbrite. Look for 'community events' in 'Chicago'. Select the first event happening tomorrow that is not promoted. Check if they have an option for 'refund policy'.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background (slightly warm off-white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFBFD")

# Status bar area at top (~88px high) - muted gray
status_bar_h = 88
draw.rectangle([(0, 0), (1440, status_bar_h)], fill="#CFCFCF")

# Subtle bottom edge for status bar
draw.line([(0, status_bar_h), (1440, status_bar_h)], fill="#B7B7B7", width=1)

# Hero/banner area (image background) below status bar
hero_top = status_bar_h
hero_bottom = 720
# base dark-brown green banner
draw.rectangle([(0, hero_top), (1440, hero_bottom)], fill="#5a3f2f")
# a subtle lighter band at the very top of the hero (visual highlight)
draw.rectangle([(0, hero_top), (1440, hero_top + 36)], fill="#6d4b38")

# Create a white rounded content card that overlaps the hero (common UI)
content_card_top = hero_bottom - 40
content_card_left = 24
content_card_right = 1440 - 24
content_card_bottom = 2760
draw.rounded_rectangle(
    [(content_card_left, content_card_top), (content_card_right, content_card_bottom)],
    radius=28, fill="#FFFFFF"
)

# Thin subtle shadow under hero/content card to separate sections
shadow_y = content_card_top + 2
draw.line([(content_card_left + 8, shadow_y), (content_card_right - 8, shadow_y)], fill="#E8E7EA", width=2)

# Organizer/profile card background (rounded, light gray) — behind profile avatar and follow button
org_card_top = 1168
org_card_bottom = 1296
org_card_left = 48
org_card_right = 1392
draw.rounded_rectangle(
    [(org_card_left, org_card_top), (org_card_right, org_card_bottom)],
    radius=18, fill="#F6F6F8"
)

# Subtle inner divider line above organizer card
draw.line([(org_card_left + 12, org_card_top), (org_card_right - 12, org_card_top)], fill="#ECEBEF", width=1)

# Section separators between major content blocks
separators = [1460, 1720, 1960, 2360]
for y in separators:
    draw.line([(48, y), (1392, y)], fill="#ECEBEF", width=1)

# Light pill/bubble background for category chips area (behind where chips are pasted)
# (Place behind detected chip positions but not drawing text)
chip_x = 48
chip_y = 2040
draw.rounded_rectangle([(chip_x, chip_y), (chip_x + 420, chip_y + 64)], radius=32, fill="#F1F2F5")

# Location section background subtle (just a faint block to anchor the section)
loc_top = 2480
loc_bottom = 2640
draw.rectangle([(48, loc_top), (1392, loc_bottom)], fill="#FFFFFF")
# separator under location section
draw.line([(48, loc_bottom + 12), (1392, loc_bottom + 12)], fill="#ECEBEF", width=1)

# Bottom sticky ticket bar
bottom_bar_top = 2760
bottom_bar_bottom = 2960
draw.rectangle([(0, bottom_bar_top), (1440, bottom_bar_bottom)], fill="#F6F4F5")

# Right-side "Get tickets" button background (rounded orange)
button_left = 880
button_right = 1392
button_top = bottom_bar_top + 24
button_bottom = bottom_bar_bottom - 16
draw.rounded_rectangle(
    [(button_left, button_top), (button_right, button_bottom)],
    radius=14, fill="#D8552B"
)

# Subtle divider line above bottom bar
draw.line([(24, bottom_bar_top), (1416, bottom_bar_top)], fill="#E6E5E9", width=1)

# Small decorative left price panel background (faint)
price_panel_left = 48
price_panel_right = 760
price_panel_top = bottom_bar_top + 20
price_panel_bottom = bottom_bar_bottom - 20
draw.rectangle([(price_panel_left, price_panel_top), (price_panel_right, price_panel_bottom)], fill="#F6F4F5")

# Final subtle vignette under hero: a very light gradient band to lead into content
for i in range(12):
    alpha_strip_y = hero_bottom - 12 + i
    # compute a slightly varying gray to simulate gradient (no alpha support, so step shades)
    shade = 244 - i  # values around light gray
    shade_hex = "#{0:02x}{0:02x}{0:02x}".format(shade)
    draw.line([(24, alpha_strip_y), (1416, alpha_strip_y)], fill=shade_hex, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1195), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1195, 1344, 1339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/01_icon_Get_tickets.png
try:
    _c1 = get_crop(1, 570, 144)
    canvas.paste(_c1, (822, 2768), _c1)
except Exception:
    pass
layout["Get_tickets"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/02_icon_Education.png
try:
    _c2 = get_crop(2, 234, 144)
    canvas.paste(_c2, (48, 2277), _c2)
except Exception:
    pass
layout["Education"] = [48, 2277, 282, 2421]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 45, 63)
    canvas.paste(_c3, (1156, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [1156, 3, 1201, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/04_icon_LVE.png
try:
    _c4 = get_crop(4, 64, 69)
    canvas.paste(_c4, (179, 1), _c4)
except Exception:
    pass
layout["LVE"] = [179, 1, 243, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/05_icon_regular.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1116, 108), _c5)
except Exception:
    pass
layout["regular"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/06_icon_Ticket_sales_end_soon.png
try:
    _c6 = get_crop(6, 547, 84)
    canvas.paste(_c6, (40, 753), _c6)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [40, 753, 587, 837]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/07_icon_LVE.png
try:
    _c7 = get_crop(7, 58, 70)
    canvas.paste(_c7, (245, 0), _c7)
except Exception:
    pass
layout["LVE"] = [245, 0, 303, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/08_icon_8.08.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (36, 108), _c8)
except Exception:
    pass
layout["8.08"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/09_icon_LVE.png
try:
    _c9 = get_crop(9, 70, 73)
    canvas.paste(_c9, (305, 0), _c9)
except Exception:
    pass
layout["LVE"] = [305, 0, 375, 73]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/10_icon_8.08.png
try:
    _c10 = get_crop(10, 62, 70)
    canvas.paste(_c10, (114, 0), _c10)
except Exception:
    pass
layout["8.08"] = [114, 0, 176, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/11_icon_regular.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1260, 108), _c11)
except Exception:
    pass
layout["regular"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/12_icon_support_as_a.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1116, 108), _c12)
except Exception:
    pass
layout["support_as_a"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 44, 60)
    canvas.paste(_c13, (1328, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [1328, 3, 1372, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/14_icon_Show_map.png
try:
    _c14 = get_crop(14, 226, 144)
    canvas.paste(_c14, (1166, 2495), _c14)
except Exception:
    pass
layout["Show_map"] = [1166, 2495, 1392, 2639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 99, 59)
    canvas.paste(_c15, (1216, 3), _c15)
except Exception:
    pass
layout["icon_15"] = [1216, 3, 1315, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/16_text_8.08.png
try:
    _c16 = get_crop(16, 94, 43)
    canvas.paste(_c16, (20, 17), _c16)
except Exception:
    pass
layout["8.08"] = [20, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/17_text_Wednesday_April_24.png
try:
    _c17 = get_crop(17, 247, 144)
    canvas.paste(_c17, (288, 1155), _c17)
except Exception:
    pass
layout["Wednesday;_April_24"] = [288, 1155, 535, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/18_text_3_00_PM.png
try:
    _c18 = get_crop(18, 209, 56)
    canvas.paste(_c18, (583, 893), _c18)
except Exception:
    pass
layout["3:00_PM"] = [583, 893, 792, 949]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/19_text_Community_Day_1_2_Off_Admission.png
try:
    _c19 = get_crop(19, 247, 144)
    canvas.paste(_c19, (288, 1155), _c19)
except Exception:
    pass
layout["Community_Day_1_2_Off_Adm"] = [288, 1155, 535, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/20_text_Nina_Salem.png
try:
    _c20 = get_crop(20, 247, 144)
    canvas.paste(_c20, (288, 1155), _c20)
except Exception:
    pass
layout["Nina_Salem"] = [288, 1155, 535, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/21_text_576_Followers.png
try:
    _c21 = get_crop(21, 247, 144)
    canvas.paste(_c21, (288, 1155), _c21)
except Exception:
    pass
layout["576_Followers"] = [288, 1155, 535, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/22_text_The_Insect_Asylum.png
try:
    _c22 = get_crop(22, 1344, 144)
    canvas.paste(_c22, (48, 1422), _c22)
except Exception:
    pass
layout["The_Insect_Asylum"] = [48, 1422, 1392, 1566]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/23_text_5_hrs.png
try:
    _c23 = get_crop(23, 112, 49)
    canvas.paste(_c23, (141, 1580), _c23)
except Exception:
    pass
layout["5_hrs"] = [141, 1580, 253, 1629]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/24_text_Refund_policy.png
try:
    _c24 = get_crop(24, 299, 63)
    canvas.paste(_c24, (138, 1685), _c24)
except Exception:
    pass
layout["Refund_policy"] = [138, 1685, 437, 1748]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/25_text_No_refunds.png
try:
    _c25 = get_crop(25, 212, 49)
    canvas.paste(_c25, (141, 1774), _c25)
except Exception:
    pass
layout["No_refunds"] = [141, 1774, 353, 1823]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/26_text_About_this_event.png
try:
    _c26 = get_crop(26, 453, 65)
    canvas.paste(_c26, (44, 1982), _c26)
except Exception:
    pass
layout["About_this_event"] = [44, 1982, 497, 2047]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/27_text_Half-off_admission_every_Wednesday_from_.png
try:
    _c27 = get_crop(27, 234, 144)
    canvas.paste(_c27, (48, 2277), _c27)
except Exception:
    pass
layout["Half-off_admission_every_"] = [48, 2277, 282, 2421]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/28_text_8.00_pm.png
try:
    _c28 = get_crop(28, 171, 60)
    canvas.paste(_c28, (1003, 2217), _c28)
except Exception:
    pass
layout["8.00_pm"] = [1003, 2217, 1174, 2277]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/29_text_Read_more.png
try:
    _c29 = get_crop(29, 234, 144)
    canvas.paste(_c29, (48, 2277), _c29)
except Exception:
    pass
layout["Read_more"] = [48, 2277, 282, 2421]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/30_text_Location.png
try:
    _c30 = get_crop(30, 244, 63)
    canvas.paste(_c30, (43, 2541), _c30)
except Exception:
    pass
layout["Location"] = [43, 2541, 287, 2604]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/31_text_The_Insect_Asylum.png
try:
    _c31 = get_crop(31, 404, 62)
    canvas.paste(_c31, (138, 2665), _c31)
except Exception:
    pass
layout["The_Insect_Asylum"] = [138, 2665, 542, 2727]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/32_text_S5_-_10.png
try:
    _c32 = get_crop(32, 196, 57)
    canvas.paste(_c32, (90, 2812), _c32)
except Exception:
    pass
layout["S5_-_$10"] = [90, 2812, 286, 2869]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_14_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-16/33_clickable_Organizer_profile_picture.png
try:
    _c33 = get_crop(33, 144, 144)
    canvas.paste(_c33, (96, 1194), _c33)
except Exception:
    pass
layout["Organizer_profile_picture"] = [96, 1194, 240, 1338]
