# page_id: page_eventbrite_c7c81d1bf6744774b99294e9f124dda3_06
# screenshot: 2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-8.png
# step_index: 6/10
# task: Open Eventbrite. Search for "Fitness". Select the events in the location "Chicago". What is the price of the first event in listing?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the mobile page

# Fill canvas background (ensure dominant white)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar (top area)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill=(200, 200, 200))  # light grey status bar
draw.line([(0, status_h - 1), (1440, status_h - 1)], fill=(170, 170, 170), width=1)

# Header area subtle background (keeps header visually distinct from status bar)
header_h = status_h + 80
draw.rectangle([(0, status_h), (1440, header_h)], fill=(250, 250, 250))

# Thick accent underline below the "Find events in..." header
underline_y = 320
draw.line([(48, underline_y), (1392, underline_y)], fill=(64, 89, 255), width=4)  # indigo/blue underline

# Card-like container behind the option icons ("Nearby" / "Online events")
card_left, card_top, card_right, card_bottom = 48, 200, 1392, 440
# subtle shadow
draw.rectangle([(card_left + 4, card_top + 6), (card_right + 4, card_bottom + 6)], fill=(245, 245, 248))
# rounded card
try:
    draw.rounded_rectangle([(card_left, card_top), (card_right, card_bottom)],
                           radius=18, fill=(250, 251, 255), outline=(230, 230, 235), width=1)
except AttributeError:
    # fallback to normal rectangle if rounded_rectangle not available
    draw.rectangle([(card_left, card_top), (card_right, card_bottom)], fill=(250, 251, 255),
                   outline=(230, 230, 235))

# Subtle horizontal separator under the card area
sep_y = card_bottom + 24
draw.line([(48, sep_y), (1392, sep_y)], fill=(235, 235, 240), width=1)

# Section divider above "Browsing in" to group sections
browsing_divider_y = 720
draw.line([(44, browsing_divider_y), (1396, browsing_divider_y)], fill=(240, 240, 245), width=1)

# A faint background band behind the selected location block to give structure
loc_band_top = 760
loc_band_bottom = 940
band_left = 44
band_right = 1396
try:
    draw.rounded_rectangle([(band_left, loc_band_top), (band_right, loc_band_bottom)],
                           radius=12, fill=(255, 255, 255), outline=None)
except AttributeError:
    draw.rectangle([(band_left, loc_band_top), (band_right, loc_band_bottom)], fill=(255, 255, 255))

# Right-side subtle circular area hint for the eventual checkmark (background only)
# (Do NOT draw the check icon itself; only provide faint circular backdrop)
circle_center = (1290 + 52 // 2, 835 + 52 // 2)  # approximate center near detected icon[0]
circle_radius = 44
draw.ellipse([(circle_center[0] - circle_radius, circle_center[1] - circle_radius),
              (circle_center[0] + circle_radius, circle_center[1] + circle_radius)],
             fill=(250, 250, 252), outline=(245, 245, 248))

# Bottom area remains clean/white for content list (no additional drawing)
# Final thin footer separator (very faint)
draw.line([(48, 2920), (1392, 2920)], fill=(245, 245, 248), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_06_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-8/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_06_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-8/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 60, 61)
    canvas.paste(_c1, (310, 4), _c1)
except Exception:
    pass
layout["icon_1"] = [310, 4, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_06_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-8/02_icon_7.10.png
try:
    _c2 = get_crop(2, 59, 63)
    canvas.paste(_c2, (180, 2), _c2)
except Exception:
    pass
layout["7.10"] = [180, 2, 239, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_06_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-8/03_icon_7.10.png
try:
    _c3 = get_crop(3, 56, 64)
    canvas.paste(_c3, (117, 1), _c3)
except Exception:
    pass
layout["7.10"] = [117, 1, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_06_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-8/04_icon_7.10.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["7.10"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_06_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-8/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 50, 57)
    canvas.paste(_c5, (248, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 6, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_06_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-8/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 71, 62)
    canvas.paste(_c6, (1213, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1213, 1, 1284, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_06_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-8/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 47, 59)
    canvas.paste(_c7, (1322, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1322, 3, 1369, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_06_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-8/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 46, 63)
    canvas.paste(_c8, (1267, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1267, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_06_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-8/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 51, 65)
    canvas.paste(_c9, (383, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 1, 434, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_06_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-8/10_text_7.10.png
try:
    _c10 = get_crop(10, 89, 41)
    canvas.paste(_c10, (22, 17), _c10)
except Exception:
    pass
layout["7.10"] = [22, 17, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_06_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-8/11_text_Find_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_06_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-8/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_06_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-8/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_06_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-8/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_06_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-8/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_06_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-8/16_text_Browsing_in.png
try:
    _c16 = get_crop(16, 228, 55)
    canvas.paste(_c16, (44, 742), _c16)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_06_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-8/17_text_New_York.png
try:
    _c17 = get_crop(17, 1440, 138)
    canvas.paste(_c17, (0, 816), _c17)
except Exception:
    pass
layout["New_York"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_06_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-8/18_text_New_York.png
try:
    _c18 = get_crop(18, 182, 50)
    canvas.paste(_c18, (44, 905), _c18)
except Exception:
    pass
layout["New_York"] = [44, 905, 226, 955]
