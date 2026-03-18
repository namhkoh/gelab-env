# page_id: page_eventbrite_47f784058c8444bd8017b372f0857efe_09
# screenshot: 2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11.png
# step_index: 9/11
# task: Open Eventbrite. Explore local events scheduled for this weekend. Select the first event from the 'Science' category. Read details of the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background (page white)
draw.rectangle([(0, 0), canvas.size], fill="#FFFFFF")

w, h = canvas.size

# Status bar area (top ~88px) - muted dark bar for time/signal area
status_bar_h = 88
draw.rectangle([(0, 0), (w, status_bar_h)], fill="#9A9A9A")
# thin divider under status bar
draw.line([(0, status_bar_h), (w, status_bar_h)], fill="#7F7F7F", width=1)

# Hero image background area (placeholder dark/soft gradient)
hero_top = status_bar_h
hero_bottom = 520
# gradient from darker to lighter to suggest photo area
top_color = (80, 80, 80)
bottom_color = (200, 200, 200)
for yy in range(hero_top, hero_bottom):
    t = (yy - hero_top) / max(1, (hero_bottom - hero_top - 1))
    r = int(top_color[0] * (1 - t) + bottom_color[0] * t)
    g = int(top_color[1] * (1 - t) + bottom_color[1] * t)
    b = int(top_color[2] * (1 - t) + bottom_color[2] * t)
    draw.line([(0, yy), (w, yy)], fill=(r, g, b))

# Subtle vignette edges to mimic photo fade (soft dark bands at sides)
vignette_color = (230, 230, 230)
band = 60
for i in range(band):
    alpha = int(40 * (1 - i / band))
    c = (200 - int(alpha/2), 200 - int(alpha/2), 200 - int(alpha/2))
    draw.rectangle([(i, hero_top), (i+1, hero_bottom)], fill=c)
    draw.rectangle([(w-1-i, hero_top), (w-i, hero_bottom)], fill=c)

# Large content card (rounded) overlapping the hero area
card_left = 36
card_right = w - 36
card_top = hero_bottom - 40   # slight overlap with hero image
card_bottom = 2660
card_radius = 28

# shadow for the card (soft, drawn as offset rectangle)
shadow_offset = 10
shadow_box = [card_left + shadow_offset, card_top + shadow_offset, card_right + shadow_offset, card_bottom + shadow_offset]
draw.rounded_rectangle(shadow_box, radius=card_radius, fill="#E9E7EA")

# main card
card_box = [card_left, card_top, card_right, card_bottom]
draw.rounded_rectangle(card_box, radius=card_radius, fill="#FFFFFF")

# Section separators (light dividers) inside the card
sep_color = "#ECE9ED"
# separator under location/refund section (approx)
sep_y1 = card_top + 560
draw.line([(card_left + 24, sep_y1), (card_right - 24, sep_y1)], fill=sep_color, width=2)

# separator under "About this event" area
sep_y2 = card_top + 1220
draw.line([(card_left + 24, sep_y2), (card_right - 24, sep_y2)], fill=sep_color, width=2)

# separator above Agenda
sep_y3 = card_top + 1760
draw.line([(card_left + 24, sep_y3), (card_right - 24, sep_y3)], fill=sep_color, width=2)

# Light rule near top of card (thin)
draw.line([(card_left + 12, card_top + 20), (card_right - 12, card_top + 20)], fill="#F2F1F3", width=1)

# Content area subtle background band for the "About" header region (very light gray)
about_band_top = card_top + 880
about_band_bottom = about_band_top + 240
draw.rectangle([(card_left + 12, about_band_top), (card_right - 12, about_band_bottom)], fill="#FFFFFF")

# Small pale background for tag container area (do NOT draw the tag text or pill itself)
# We intentionally do not draw the pill shape that would duplicate detected tag elements.
# Instead provide a faint backdrop for the section area (keeps visual separation).
tag_back_left = card_left + 12
tag_back_right = card_left + 340
tag_back_top = card_top + 920
tag_back_bottom = tag_back_top + 72
draw.rectangle([(tag_back_left, tag_back_top), (tag_back_right, tag_back_bottom)], fill="#FAFAFB")

# Footer area (bottom persistent bar) - light neutral background
footer_top = 2680
draw.rectangle([(0, footer_top), (w, h)], fill="#F6F4F5")
# thin divider line above footer
draw.line([(0, footer_top), (w, footer_top)], fill="#E6E3E6", width=2)

# Left area of footer where "Free" label sits - subtle separation (do not draw the text)
footer_left_box = [24, footer_top + 24, int(w * 0.55) - 12, h - 24]
draw.rectangle(footer_left_box, fill="#F6F4F5")

# Right side background area behind the CTA will remain untouched so the auto-pasted "Get tickets" button can be placed.
# Provide a slight rounded highlight behind where the CTA might rest (but avoid overlapping exact button bounds).
cta_bg_left = int(w * 0.58)
cta_bg_top = footer_top + 20
cta_bg_right = w - 24
cta_bg_bottom = h - 28
# Draw only a subtle rounded container (very light) to frame the area, but do not mimic the actual CTA color/shape.
draw.rounded_rectangle([cta_bg_left, cta_bg_top, cta_bg_right, cta_bg_bottom], radius=14, fill="#FFFFFF")

# Decorative thin horizontal guide lines inside card to suggest text groupings (no actual text drawn)
guide_x1 = card_left + 36
guide_x2 = card_right - 36
for y in (card_top + 160, card_top + 280, card_top + 420, card_top + 640, card_top + 760, card_top + 980, card_top + 1100):
    draw.line([(guide_x1, y), (guide_x2, y)], fill="#FBFBFC", width=1)

# final subtle vignette at bottom of content card to anchor composition
vbot_top = card_bottom - 80
for i in range(80):
    alpha = int(12 * (1 - i / 80))
    c = (240 - alpha, 240 - alpha, 241 - alpha)
    draw.line([(card_left + 6, vbot_top + i), (card_right - 6, vbot_top + i)], fill=c)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/00_icon_Get_tickets.png
try:
    _c0 = get_crop(0, 570, 144)
    canvas.paste(_c0, (822, 2768), _c0)
except Exception:
    pass
layout["Get_tickets"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/01_icon_More.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1116, 108), _c1)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/02_icon_Share.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1260, 108), _c2)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/03_icon_Going_fast.png
try:
    _c3 = get_crop(3, 334, 87)
    canvas.paste(_c3, (41, 753), _c3)
except Exception:
    pass
layout["Going_fast"] = [41, 753, 375, 840]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/04_icon_7.59.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (36, 108), _c4)
except Exception:
    pass
layout["7.59"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/05_icon_Science_Technology.png
try:
    _c5 = get_crop(5, 234, 144)
    canvas.paste(_c5, (48, 2187), _c5)
except Exception:
    pass
layout["Science_&_Technology"] = [48, 2187, 282, 2331]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/06_icon_7.59.png
try:
    _c6 = get_crop(6, 65, 70)
    canvas.paste(_c6, (179, 1), _c6)
except Exception:
    pass
layout["7.59"] = [179, 1, 244, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/07_icon_7.59.png
try:
    _c7 = get_crop(7, 62, 71)
    canvas.paste(_c7, (114, 0), _c7)
except Exception:
    pass
layout["7.59"] = [114, 0, 176, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 54, 63)
    canvas.paste(_c8, (1318, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1318, 1, 1372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 56, 69)
    canvas.paste(_c9, (247, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [247, 1, 303, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 66, 70)
    canvas.paste(_c10, (308, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [308, 1, 374, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/11_icon_9_30AM.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1116, 108), _c11)
except Exception:
    pass
layout["9:30AM"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 63, 60)
    canvas.paste(_c12, (1216, 3), _c12)
except Exception:
    pass
layout["icon_12"] = [1216, 3, 1279, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 62, 61)
    canvas.paste(_c13, (1252, 2), _c13)
except Exception:
    pass
layout["icon_13"] = [1252, 2, 1314, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 52, 69)
    canvas.paste(_c14, (382, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [382, 1, 434, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/15_icon_The_organizer_will_review_refund_request.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 1277), _c15)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1277, 1392, 1421]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/16_text_7.59.png
try:
    _c16 = get_crop(16, 89, 43)
    canvas.paste(_c16, (22, 17), _c16)
except Exception:
    pass
layout["7.59"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/17_text_Area_Bioengineering_Symposium.png
try:
    _c17 = get_crop(17, 1344, 144)
    canvas.paste(_c17, (48, 1277), _c17)
except Exception:
    pass
layout["Area_Bioengineering_Sympo"] = [48, 1277, 1392, 1421]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/18_text_BABS.png
try:
    _c18 = get_crop(18, 245, 84)
    canvas.paste(_c18, (46, 1109), _c18)
except Exception:
    pass
layout["[BABS]"] = [46, 1109, 291, 1193]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/19_text_About_this_event.png
try:
    _c19 = get_crop(19, 454, 61)
    canvas.paste(_c19, (45, 1840), _c19)
except Exception:
    pass
layout["About_this_event"] = [45, 1840, 499, 1901]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/20_text_Join_us_for_an_in-depth_dive_into_the_bi.png
try:
    _c20 = get_crop(20, 234, 144)
    canvas.paste(_c20, (48, 2187), _c20)
except Exception:
    pass
layout["Join_us_for_an_in-depth_d"] = [48, 2187, 282, 2331]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/21_text_Areal.png
try:
    _c21 = get_crop(21, 105, 43)
    canvas.paste(_c21, (457, 2133), _c21)
except Exception:
    pass
layout["Areal"] = [457, 2133, 562, 2176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/22_text_Read_more.png
try:
    _c22 = get_crop(22, 234, 144)
    canvas.paste(_c22, (48, 2187), _c22)
except Exception:
    pass
layout["Read_more"] = [48, 2187, 282, 2331]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/23_text_Agenda.png
try:
    _c23 = get_crop(23, 229, 75)
    canvas.paste(_c23, (42, 2449), _c23)
except Exception:
    pass
layout["Agenda"] = [42, 2449, 271, 2524]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/24_text_Free.png
try:
    _c24 = get_crop(24, 110, 55)
    canvas.paste(_c24, (89, 2816), _c24)
except Exception:
    pass
layout["Free"] = [89, 2816, 199, 2871]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_09_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-11/25_clickable_Click_to_view_the_detailed_agenda_of_the.png
try:
    _c25 = get_crop(25, 1440, 372)
    canvas.paste(_c25, (0, 2588), _c25)
except Exception:
    pass
layout["Click_to_view_the_detaile"] = [0, 2588, 1440, 2960]
