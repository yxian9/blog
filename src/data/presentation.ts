type Social = {
  label: string;
  link: string;
};

type Presentation = {
  mail: string;
  title: string;
  description: string;
  socials: Social[];
  profile?: string;
};

const presentation: Presentation = {
  mail: "My Blog",
  title: "Hi, I’m Yu X 👋",
  profile: "/profile.webp",
  description:
    "Hello, I am a *computational biologist* working on genomics and epigenomics.  I'm interested in developing scalable computational workflows for analyzing big data genomics.",
  socials: [
    {
      label: "Linkedin",
      link: "https://www.linkedin.com/in/yu--xiang/",
    },
    {
      label: "Google Scholar",
      link: "https://scholar.google.com/citations?user=ZGc5hzsAAAAJ&hl=en",
    },
  ],
};

export default presentation;
