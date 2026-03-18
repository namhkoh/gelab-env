# page_id: page_eventbrite_b2798d8b10cc4118ab8cf6648f8a4077_07
# screenshot: 2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-9.png
# step_index: 7/12
# task: Open Eventbrite. Search Music event in New York. Select the first one. Record its location and time in Google Keep Notes. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar background
draw.rectangle((0, 0, 1440, 72), fill=(153, 153, 153))

# Subtle divider under status bar
draw.line((0, 72, 1440, 72), fill=(128, 128, 128), width=1)

# Overall page background (explicit, matches screenshot's dominant white)
draw.rectangle((0, 72, 1440, 2960), fill=(255, 255, 255))

# Purple accent underline beneath the "Find events in..." header
# left/right margins match UI (48px)
underline_y = 393
draw.line((48, underline_y, 1392, underline_y), fill=(63, 81, 181), width=6)

# Thin light section separator between header area and list content
draw.line((48, 460, 1392, 460), fill=(230, 230, 235), width=1)

# Light rounded card background for the "Browsing in / Online events" block
card_left = 36
card_top = 720
card_right = 1404
card_bottom = 940
draw.rounded_rectangle((card_left, card_top, card_right, card_bottom),
                       radius=18,
                       fill=(250, 250, 252),
                       outline=None)

# Subtle shadow line at top of that card to separate from content above
draw.line((card_left + 8, card_top + 1, card_right - 8, card_top + 1),
          fill=(235, 235, 238), width=1)

# Smaller separator under the "Nearby" list item area
draw.line((48, 560, 1392, 560), fill=(240, 240, 242), width=1)

# Right-side faint circular selection background (soft halo)
# Drawn as a background accent, positioned away from known icon crops where possible.
# Use a very subtle fill so the actual check icon (pasted later) remains primary.
halo_center = (1291 + 51, 837 + 54)  # center based on detected icon box; OK to underlay
halo_radius = 68
draw.ellipse((halo_center[0] - halo_radius, halo_center[1] - halo_radius,
              halo_center[0] + halo_radius, halo_center[1] + halo_radius),
             fill=(250, 250, 252))

# Bottom subtle divider to indicate end of header content area
draw.line((48, 820, 1392, 820), fill=(245, 245, 247), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_07_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-9/00_icon_9_19.png
try:
    _c0 = get_crop(0, 58, 63)
    canvas.paste(_c0, (179, 1), _c0)
except Exception:
    pass
layout["9:19"] = [179, 1, 237, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_07_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-9/01_icon_9_19.png
try:
    _c1 = get_crop(1, 51, 63)
    canvas.paste(_c1, (117, 2), _c1)
except Exception:
    pass
layout["9:19"] = [117, 2, 168, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_07_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-9/02_icon_9_19.png
try:
    _c2 = get_crop(2, 168, 168)
    canvas.paste(_c2, (0, 72), _c2)
except Exception:
    pass
layout["9:19"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_07_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-9/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 56, 62)
    canvas.paste(_c3, (246, 1), _c3)
except Exception:
    pass
layout["icon_3"] = [246, 1, 302, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_07_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-9/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 54, 63)
    canvas.paste(_c4, (315, 1), _c4)
except Exception:
    pass
layout["icon_4"] = [315, 1, 369, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_07_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-9/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 94, 61)
    canvas.paste(_c5, (1213, 1), _c5)
except Exception:
    pass
layout["icon_5"] = [1213, 1, 1307, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_07_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-9/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 50, 57)
    canvas.paste(_c6, (1320, 4), _c6)
except Exception:
    pass
layout["icon_6"] = [1320, 4, 1370, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_07_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-9/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 103, 108)
    canvas.paste(_c7, (1291, 837), _c7)
except Exception:
    pass
layout["icon_7"] = [1291, 837, 1394, 945]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_07_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-9/08_text_9_19.png
try:
    _c8 = get_crop(8, 94, 45)
    canvas.paste(_c8, (17, 15), _c8)
except Exception:
    pass
layout["9:19"] = [17, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_07_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-9/09_text_Find_events_in..png
try:
    _c9 = get_crop(9, 1344, 129)
    canvas.paste(_c9, (48, 264), _c9)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_07_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-9/10_text_Nearby.png
try:
    _c10 = get_crop(10, 415, 114)
    canvas.paste(_c10, (48, 465), _c10)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_07_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-9/11_text_Current_location.png
try:
    _c11 = get_crop(11, 415, 114)
    canvas.paste(_c11, (48, 465), _c11)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_07_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-9/12_text_Browsing_in.png
try:
    _c12 = get_crop(12, 228, 55)
    canvas.paste(_c12, (44, 742), _c12)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_07_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-9/13_text_Online_events.png
try:
    _c13 = get_crop(13, 1440, 138)
    canvas.paste(_c13, (0, 816), _c13)
except Exception:
    pass
layout["Online_events"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_07_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-9/14_text_Virtual_attendance.png
try:
    _c14 = get_crop(14, 1440, 138)
    canvas.paste(_c14, (0, 816), _c14)
except Exception:
    pass
layout["Virtual_attendance"] = [0, 816, 1440, 954]
