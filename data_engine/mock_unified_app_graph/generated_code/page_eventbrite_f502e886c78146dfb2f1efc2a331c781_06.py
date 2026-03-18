# page_id: page_eventbrite_f502e886c78146dfb2f1efc2a331c781_06
# screenshot: 2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-8.png
# step_index: 6/18
# task: Open Eventbrite. Search for 'music festival' in San Francisco. Set date available from April 30 to May 3. How many events are listed?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for 1440x2960 canvas.
# Available: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Overall background (keep pristine white base)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar area (top ~72px) - subtle gray
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill=(200, 200, 200))

# Thin darker top hairline for separation
draw.line([(0, status_h), (1440, status_h)], fill=(180, 180, 180), width=1)

# Header / toolbar area (under status bar) - white with subtle shadow line
header_top = status_h
header_bottom = 160
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))
draw.line([(24, header_bottom), (1416, header_bottom)], fill=(230, 230, 230), width=1)

# Large search/title area background subtle (behind the "Find events in..." text)
# Keep it very light so pasted text remains clear; no text drawn here.
search_bg_top = header_bottom + 14
search_bg_bottom = search_bg_top + 140
draw.rectangle([(0, search_bg_top), (1440, search_bg_bottom)], fill=(255, 255, 255))

# Underline / divider below the search title (accent thin rule)
divider_y = search_bg_bottom + 10
draw.line([(48, divider_y), (1392, divider_y)], fill=(200, 193, 208), width=2)

# Card background for the "options" row (Nearby / Online events) - subtle cool tint
# Use a rounded rectangle spanning across the area without drawing icons/text.
options_card = [36, divider_y + 32, 1404, divider_y + 180]
draw.rounded_rectangle(options_card, radius=18, fill=(247, 250, 255), outline=(230, 235, 250), width=1)

# Separator below options card
sep_y = options_card[3] + 18
draw.line([(24, sep_y), (1416, sep_y)], fill=(240, 238, 243), width=1)

# "Browsing in" section card area (large white card background to anchor location selection)
browsing_card = [24, sep_y + 28, 1416, sep_y + 200]
# subtle shadow effect (top and bottom faint lines)
draw.rectangle([(browsing_card[0], browsing_card[1]), (browsing_card[2], browsing_card[3])], fill=(255, 255, 255))
draw.line([(browsing_card[0]+8, browsing_card[1]), (browsing_card[2]-8, browsing_card[1])], fill=(245, 243, 247), width=1)
draw.line([(browsing_card[0]+8, browsing_card[3]), (browsing_card[2]-8, browsing_card[3])], fill=(245, 243, 247), width=1)

# Accent circular background on the right side to echo the selection affordance (no checkmark drawn)
# Keep it as a pale, subtle circle so it won't conflict with pasted check icon.
accent_circle_center = (1320, browsing_card[1] + 84)
accent_radius = 46
draw.ellipse(
    [(accent_circle_center[0] - accent_radius, accent_circle_center[1] - accent_radius),
     (accent_circle_center[0] + accent_radius, accent_circle_center[1] + accent_radius)],
    fill=(250, 249, 252), outline=(235, 233, 239)
)

# Thin section divider below browsing card to separate from main content area
after_browsing_y = browsing_card[3] + 18
draw.line([(24, after_browsing_y), (1416, after_browsing_y)], fill=(240, 239, 243), width=1)

# Large empty content area (keeps white) with a faint guide band near top of content for possible image posts
content_guide_top = after_browsing_y + 28
content_guide_bottom = content_guide_top + 320
draw.rectangle([(36, content_guide_top), (1404, content_guide_bottom)], fill=(250, 250, 250))
# Light border to indicate a content card slot (no inner visuals)
draw.rectangle([(36, content_guide_top), (1404, content_guide_bottom)], outline=(235, 235, 238), width=1)

# Another subtle divider for continued list area
list_start_y = content_guide_bottom + 24
draw.line([(24, list_start_y), (1416, list_start_y)], fill=(245, 244, 246), width=1)

# Footer spacing guideline (very faint)
footer_guideline_y = 2800
draw.line([(0, footer_guideline_y), (1440, footer_guideline_y)], fill=(250, 250, 250), width=1)

# End of structural drawing. UI content (icons/text) will be pasted on top of these backgrounds.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_06_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-8/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_06_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-8/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 60, 60)
    canvas.paste(_c1, (310, 4), _c1)
except Exception:
    pass
layout["icon_1"] = [310, 4, 370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_06_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-8/02_icon_7.18.png
try:
    _c2 = get_crop(2, 59, 63)
    canvas.paste(_c2, (180, 2), _c2)
except Exception:
    pass
layout["7.18"] = [180, 2, 239, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_06_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-8/03_icon_7.18.png
try:
    _c3 = get_crop(3, 57, 64)
    canvas.paste(_c3, (116, 1), _c3)
except Exception:
    pass
layout["7.18"] = [116, 1, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_06_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-8/04_icon_7.18.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["7.18"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_06_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-8/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 49, 57)
    canvas.paste(_c5, (249, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [249, 6, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_06_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-8/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 46, 59)
    canvas.paste(_c6, (1322, 3), _c6)
except Exception:
    pass
layout["icon_6"] = [1322, 3, 1368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_06_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-8/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 60, 64)
    canvas.paste(_c7, (1213, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1213, 1, 1273, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_06_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-8/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 43, 63)
    canvas.paste(_c8, (1270, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1270, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_06_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-8/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 51, 65)
    canvas.paste(_c9, (383, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 1, 434, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_06_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-8/10_icon_7.18.png
try:
    _c10 = get_crop(10, 93, 64)
    canvas.paste(_c10, (14, 1), _c10)
except Exception:
    pass
layout["7.18"] = [14, 1, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_06_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-8/11_icon_Nearby.png
try:
    _c11 = get_crop(11, 415, 114)
    canvas.paste(_c11, (48, 465), _c11)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_06_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-8/12_text_Find_events_in..png
try:
    _c12 = get_crop(12, 1344, 129)
    canvas.paste(_c12, (48, 264), _c12)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_06_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-8/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_06_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-8/14_text_Virtual_attendance.png
try:
    _c14 = get_crop(14, 452, 114)
    canvas.paste(_c14, (511, 465), _c14)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_06_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-8/15_text_Browsing_in.png
try:
    _c15 = get_crop(15, 228, 55)
    canvas.paste(_c15, (44, 742), _c15)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_06_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-8/16_text_Los_Angeles.png
try:
    _c16 = get_crop(16, 1440, 138)
    canvas.paste(_c16, (0, 816), _c16)
except Exception:
    pass
layout["Los_Angeles"] = [0, 816, 1440, 954]
