# page_id: page_eventbrite_3ce6196f48694f74bf7d05dc71840c63_09
# screenshot: 2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11.png
# step_index: 9/9
# task: Open Eventbrite. Search for 'coding workshop'. Sort the results by date. Where is the location of the soonest event that is not promoted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (1440x2960). Draw only background and structural elements.
# Do not draw any detected icons/text/buttons.

# Overall page background (very light off-white)
draw.rectangle([(0, 0), (1440, 2960)], fill=(246, 245, 248))

# Status bar area (top ~72px) - muted gray strip for time/signal background
draw.rectangle([(0, 0), (1440, 72)], fill=(189, 189, 189))

# Hero/banner background area (under status bar) - use soft banded fills to suggest an image banner
draw.rectangle([(0, 72), (1440, 180)], fill=(225, 229, 235))
draw.rectangle([(0, 180), (1440, 320)], fill=(210, 218, 228))
draw.rectangle([(0, 320), (1440, 472)], fill=(230, 232, 237))

# Subtle translucent overlay strip near top of banner for toolbar region (keeps icons visible)
draw.rectangle([(0, 72), (1440, 120)], fill=(255, 255, 255, 40))

# Main content background card under the banner (white content area)
content_top = 472
content_left = 24
content_right = 1440 - 24
# Large white sheet where page content sits
draw.rounded_rectangle([(content_left, content_top), (content_right, 2200)], radius=8, fill=(255, 255, 255))

# Organizer inline card background (rounded light panel) behind avatar, name and follow button
org_card_top = 1000
org_card_bottom = 1164
org_card_left = 48
org_card_right = 1392
draw.rounded_rectangle([(org_card_left, org_card_top), (org_card_right, org_card_bottom)], radius=22, fill=(247, 246, 249))

# Thin separator/divider between sections
def hor_div(y):
    draw.line([(48, y), (1392, y)], fill=(226, 224, 230), width=2)

hor_div(1288)   # divider under event info / refund area
hor_div(1528)   # divider under "About this event" section
hor_div(2048)   # divider above location card

# Location card area (white block for location details)
loc_top = 2080
loc_bottom = 2560
draw.rounded_rectangle([(48, loc_top), (1392, loc_bottom)], radius=8, fill=(255, 255, 255))

# Subtle left gutter vertical rule to visually separate content from page edge
draw.rectangle([(24, content_top), (32, 2200)], fill=(246, 245, 248))

# Floating subtle shadow line above the bottom action bar
draw.line([(0, 2696), (1440, 2696)], fill=(230, 227, 232), width=2)

# Bottom sticky action bar background (light neutral) - leave space for the actual "Get tickets" button to be pasted
footer_top = 2700
footer_bottom = 2960
draw.rectangle([(0, footer_top), (1440, footer_bottom)], fill=(250, 249, 250))

# Slightly darker inset panel on left side of footer (area where price will appear)
draw.rectangle([(24, footer_top + 20), (480, footer_bottom - 20)], fill=(255, 255, 255))
draw.rectangle([(24, footer_top + 20), (480, footer_bottom - 20)], outline=(232, 229, 233), width=1)

# Right side reserved for the "Get tickets" control - keep background neutral (do not draw button)
# Draw an ambient rounded placeholder area (no text/icon) to indicate reserved space
reserved_right = (520, footer_top + 16, 1416, footer_bottom - 16)
draw.rounded_rectangle([ (reserved_right[0], reserved_right[1]), (reserved_right[2], reserved_right[3]) ],
                       radius=10, fill=(255,255,255,0), outline=(235, 233, 238), width=1)

# Small divider lines to structure content columns inside main content area
# e.g., separate metadata icons from details (purely decorative)
draw.line([(48, 1320), (1392, 1320)], fill=(245, 244, 247), width=1)
draw.line([(48, 1680), (1392, 1680)], fill=(245, 244, 247), width=1)

# Soft shadow under the hero/banner to separate from content
draw.rectangle([(0, 468), (1440, 476)], fill=(236, 234, 240))

# End of structural/background drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1068), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1068, 1344, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/01_icon_Get_tickets.png
try:
    _c1 = get_crop(1, 570, 144)
    canvas.paste(_c1, (822, 2768), _c1)
except Exception:
    pass
layout["Get_tickets"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/02_icon_More.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1116, 108), _c2)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/03_icon_Share.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1260, 108), _c3)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/04_icon_7.26.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (36, 108), _c4)
except Exception:
    pass
layout["7.26"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/05_icon_7.26.png
try:
    _c5 = get_crop(5, 66, 71)
    canvas.paste(_c5, (178, 0), _c5)
except Exception:
    pass
layout["7.26"] = [178, 0, 244, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/06_icon_Give_the_gift_of_education_to_someone_yo.png
try:
    _c6 = get_crop(6, 234, 144)
    canvas.paste(_c6, (48, 2145), _c6)
except Exception:
    pass
layout["Give_the_gift_of_educatio"] = [48, 2145, 282, 2289]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/07_icon_7.26.png
try:
    _c7 = get_crop(7, 64, 70)
    canvas.paste(_c7, (113, 0), _c7)
except Exception:
    pass
layout["7.26"] = [113, 0, 177, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 58, 71)
    canvas.paste(_c8, (245, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [245, 0, 303, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 72, 70)
    canvas.paste(_c9, (305, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [305, 0, 377, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 57, 65)
    canvas.paste(_c10, (1317, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1317, 0, 1374, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 100, 64)
    canvas.paste(_c11, (1214, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1214, 0, 1314, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/12_icon_Show_map.png
try:
    _c12 = get_crop(12, 226, 144)
    canvas.paste(_c12, (1166, 2363), _c12)
except Exception:
    pass
layout["Show_map"] = [1166, 2363, 1392, 2507]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/13_icon_PRNSA_Field_Institute_Gift_Certificates.png
try:
    _c13 = get_crop(13, 456, 144)
    canvas.paste(_c13, (288, 1028), _c13)
except Exception:
    pass
layout["PRNSA_Field_Institute_Gif"] = [288, 1028, 744, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 52, 70)
    canvas.paste(_c14, (382, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [382, 0, 434, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/15_text_ashore.png
try:
    _c15 = get_crop(15, 69, 25)
    canvas.paste(_c15, (127, 1066), _c15)
except Exception:
    pass
layout["ashore"] = [127, 1066, 196, 1091]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/16_text_PRNSA_Field_Institute.png
try:
    _c16 = get_crop(16, 456, 144)
    canvas.paste(_c16, (288, 1028), _c16)
except Exception:
    pass
layout["PRNSA_Field_Institute"] = [288, 1028, 744, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/17_text_159_Followers.png
try:
    _c17 = get_crop(17, 456, 144)
    canvas.paste(_c17, (288, 1028), _c17)
except Exception:
    pass
layout["159_Followers"] = [288, 1028, 744, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/18_text_Point_Reyes_National_Seashore.png
try:
    _c18 = get_crop(18, 1344, 144)
    canvas.paste(_c18, (48, 1295), _c18)
except Exception:
    pass
layout["Point_Reyes_National_Seas"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/19_text_2555_days_3_hrs.png
try:
    _c19 = get_crop(19, 350, 66)
    canvas.paste(_c19, (138, 1449), _c19)
except Exception:
    pass
layout["2555_days_3_hrs"] = [138, 1449, 488, 1515]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/20_text_Refund_policy.png
try:
    _c20 = get_crop(20, 300, 63)
    canvas.paste(_c20, (138, 1558), _c20)
except Exception:
    pass
layout["Refund_policy"] = [138, 1558, 438, 1621]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/21_text_The_organizer_will_review_refund_request.png
try:
    _c21 = get_crop(21, 1344, 144)
    canvas.paste(_c21, (48, 1295), _c21)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/22_text_Location.png
try:
    _c22 = get_crop(22, 246, 64)
    canvas.paste(_c22, (41, 2406), _c22)
except Exception:
    pass
layout["Location"] = [41, 2406, 287, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/23_text_Point_Reyes_National_Seashore.png
try:
    _c23 = get_crop(23, 234, 144)
    canvas.paste(_c23, (48, 2145), _c23)
except Exception:
    pass
layout["Point_Reyes_National_Seas"] = [48, 2145, 282, 2289]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/24_text_Point_Reyes_National_Seashore_1_Bear_Val.png
try:
    _c24 = get_crop(24, 570, 144)
    canvas.paste(_c24, (822, 2768), _c24)
except Exception:
    pass
layout["Point_Reyes_National_Seas"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/25_text_Point.png
try:
    _c25 = get_crop(25, 119, 52)
    canvas.paste(_c25, (1217, 2602), _c25)
except Exception:
    pass
layout["Point"] = [1217, 2602, 1336, 2654]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/26_text_S65.png
try:
    _c26 = get_crop(26, 103, 61)
    canvas.paste(_c26, (89, 2811), _c26)
except Exception:
    pass
layout["S65"] = [89, 2811, 192, 2872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/27_text_S360.png
try:
    _c27 = get_crop(27, 130, 57)
    canvas.paste(_c27, (213, 2812), _c27)
except Exception:
    pass
layout["S360"] = [213, 2812, 343, 2869]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_09_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-11/28_clickable_Organizer_profile_picture.png
try:
    _c28 = get_crop(28, 144, 144)
    canvas.paste(_c28, (96, 1067), _c28)
except Exception:
    pass
layout["Organizer_profile_picture"] = [96, 1067, 240, 1211]
