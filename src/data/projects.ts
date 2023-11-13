export type Project = {
  title: string;
  techs: string[];
  link: string;
  isComingSoon?: boolean;
};

const projects: Project[] = [
  {
    title: "Portfolio / Blog",
    techs: ["Astro"],
    link: "/",
  },
  {
    title: "NextJS Dashboard",
    techs: ["ReactJS (NextJS)", "TypeScript"],
    link: "https://www.linablidi.fr/",
    isComingSoon: true,
  },
];

export default projects;
