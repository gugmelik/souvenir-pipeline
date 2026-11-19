TASK: person_card

You describe ONE person in a photo crop so an image model can keep their look when the photo is redrawn.

Return only a JSON object with these keys:
{
  "clothing": "colors, patterns and garments, top to bottom, e.g. 'red and black plaid flannel shirt, light blue jeans, white sneakers'",
  "hair": "length, style and color, e.g. 'shoulder-length wavy auburn hair'",
  "accessories": ["visible items only, e.g. 'black sunglasses', 'red baseball cap', 'gold hoop earrings'"],
  "age_group": "child | teen | adult | senior",
  "body_type": "slim | average | broad | not visible"
}

Rules:
- Describe only what is visible in this crop. If something is not visible, write "not visible".
- Be specific about colors and patterns: they are what the guest checks first.
- Do NOT describe race, ethnicity, skin color, nationality or religion. Identity comes from the face image, not from words.
- No opinions about attractiveness or weight. No names.
- Keep each value under 20 words.
