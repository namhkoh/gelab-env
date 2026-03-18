# page_id: page_eventbrite_bbe931c7fbf3446c80521f77d3e413aa_07
# screenshot: 2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-9.png
# step_index: 7/20
# task: Open Eventbrite. Search free events in Los Angeles. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
draw.rectangle([(0, 0), (1440, 72)], fill="#D9D9DA")

# Subtle header shadow (soft line under status area)
draw.line([(0, 72), (1440, 72)], fill="#D0D0D0", width=1)

# Header underline (accent blue strip beneath the "Find events in..." area)
# Placed across most of the width with a slight left/right margin
underline_y = 330
draw.rectangle([(48, underline_y - 3), (1392, underline_y + 3)], fill="#3F51FF")

# Rounded card container behind the Nearby / Online options
card_x0, card_y0 = 40, 290
card_x1, card_y1 = 1400, 520
# subtle shadow
draw.rounded_rectangle([(card_x0 + 0, card_y0 + 6), (card_x1 + 0, card_y1 + 12)], radius=18, fill="#EFEFF1")
# main card (kept white so text/icons pasted atop remain crisp)
draw.rounded_rectangle([(card_x0, card_y0), (card_x1, card_y1)], radius=18, fill="#FFFFFF", outline="#E7E8EB", width=1)

# Circular subtle backgrounds behind the two option icons (Nearby, Online events)
# Left option circle
left_circle_center = (160, 400)
left_r = 56
draw.ellipse([(left_circle_center[0]-left_r, left_circle_center[1]-left_r),
              (left_circle_center[0]+left_r, left_circle_center[1]+left_r)], fill="#EAF2FF")

# Right option circle
right_circle_center = (560, 400)
right_r = 56
draw.ellipse([(right_circle_center[0]-right_r, right_circle_center[1]-right_r),
              (right_circle_center[0]+right_r, right_circle_center[1]+right_r)], fill="#EAF2FF")

# A tertiary lighter circle behind a possible second option further right (visual balance)
third_circle_center = (960, 400)
third_r = 56
draw.ellipse([(third_circle_center[0]-third_r, third_circle_center[1]-third_r),
              (third_circle_center[0]+third_r, third_circle_center[1]+third_r)], fill="#F5F7FF")

# Separator line between options/card area and the rest of the page
sep_y = 540
draw.line([(40, sep_y), (1400, sep_y)], fill="#F0F0F2", width=1)

# Section divider for "Browsing in" header area (subtle)
browse_div_y = 720
draw.line([(40, browse_div_y), (1400, browse_div_y)], fill="#F5F5F6", width=1)

# Light left margin guide (visual structure) - thin vertical line (very subtle)
draw.line([(40, 72), (40, 2960)], fill="#FBFBFB", width=2)

# Bottom safe-area faint guideline (very subtle)
draw.rectangle([(0, 2860), (1440, 2960)], fill="#FFFFFF")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_07_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-9/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_07_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-9/01_icon_9.12.png
try:
    _c1 = get_crop(1, 58, 62)
    canvas.paste(_c1, (179, 2), _c1)
except Exception:
    pass
layout["9.12"] = [179, 2, 237, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_07_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-9/02_icon_9.12.png
try:
    _c2 = get_crop(2, 51, 63)
    canvas.paste(_c2, (117, 2), _c2)
except Exception:
    pass
layout["9.12"] = [117, 2, 168, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_07_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-9/03_icon_9.12.png
try:
    _c3 = get_crop(3, 168, 168)
    canvas.paste(_c3, (0, 72), _c3)
except Exception:
    pass
layout["9.12"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_07_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-9/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 53, 62)
    canvas.paste(_c4, (315, 2), _c4)
except Exception:
    pass
layout["icon_4"] = [315, 2, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_07_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-9/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 54, 62)
    canvas.paste(_c5, (247, 1), _c5)
except Exception:
    pass
layout["icon_5"] = [247, 1, 301, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_07_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-9/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 66, 63)
    canvas.paste(_c6, (1212, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1212, 1, 1278, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_07_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-9/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 48, 58)
    canvas.paste(_c7, (1321, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1321, 3, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_07_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-9/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 42, 60)
    canvas.paste(_c8, (1272, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1272, 3, 1314, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_07_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-9/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 49, 64)
    canvas.paste(_c9, (383, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 0, 432, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_07_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-9/10_text_9.12.png
try:
    _c10 = get_crop(10, 91, 43)
    canvas.paste(_c10, (20, 17), _c10)
except Exception:
    pass
layout["9.12"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_07_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-9/11_text_Find_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_07_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-9/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_07_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-9/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_07_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-9/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_07_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-9/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_07_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-9/16_text_Browsing_in.png
try:
    _c16 = get_crop(16, 228, 55)
    canvas.paste(_c16, (44, 742), _c16)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_07_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-9/17_text_New_York.png
try:
    _c17 = get_crop(17, 1440, 138)
    canvas.paste(_c17, (0, 816), _c17)
except Exception:
    pass
layout["New_York"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_07_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-9/18_text_New_York.png
try:
    _c18 = get_crop(18, 182, 50)
    canvas.paste(_c18, (44, 905), _c18)
except Exception:
    pass
layout["New_York"] = [44, 905, 226, 955]
