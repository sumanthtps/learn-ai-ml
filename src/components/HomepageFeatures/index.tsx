import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Structured Learning Path',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        Step-by-step notes covering Python, Data Structures, Machine Learning,
        Deep Learning, PyTorch, FastAPI, and MLOps — from fundamentals to
        industry-ready practices.
      </>
    ),
  },
  {
    title: 'Code-First Notes',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Every concept is backed by working code examples, visual explanations,
        and common interview questions so you understand not just <em>how</em>,
        but <em>why</em>.
      </>
    ),
  },
  {
    title: 'Real-World Projects',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        Build CNNs, RNNs, LSTMs, transfer learning pipelines, REST APIs with
        FastAPI, and end-to-end MLOps systems — the same stack used in
        production AI teams.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
