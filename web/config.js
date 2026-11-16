// Settings you can change without touching the game code.
export const CONFIG = {
  // Link shown on the home screen ("Watch the videos"). Leave empty to hide the button.
  youtube: '',

  // Ads (Adsterra banners). Nothing loads while `on` is false or a key is empty.
  // In the Adsterra dashboard, create one 320x50 banner and one 300x250 banner, then paste their keys here.
  // `host` is the start of the script address in the code Adsterra gives you (before "/<key>/invoke.js").
  ads: {
    on: false,
    host: 'https://www.highperformanceformat.com',
    top: { key: '', w: 320, h: 50 },     // at the top of the screen during a round
    card: { key: '', w: 300, h: 250 },   // under the result card
  },
};
